#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TLatex.h>
#include <TROOT.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <sys/stat.h>
#include <ctime>
#include <cmath>

using std::cout;
using std::endl;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0, 10, 7, 2, 6, 3, 8, 9, 11, 4, 5, 1};
const int PULSE_THRESHOLD = 30;     // ADC threshold for pulse detection
const int EVBF_THRESHOLD = 1000;    // Beam on if channel 22 > this (ADC)
const double BEAM_TIME_CORRECTION = 2.2; // Beam delay after EVBF (µs)
const int ADCSIZE = 45;                 // Number of ADC samples per waveform
const std::string OUTPUT_DIR = "output"; // Output directory
const double FIT_MIN = 1.0; // Fit range min (µs)
const double FIT_MAX = 10.0; // Fit range max (µs)

// Event selection thresholds
const double MUON_ENERGY_THRESHOLD = 50; // Min PMT energy for muon (p.e.)
const std::vector<double> SIDE_SIPM_THRESHOLDS = {750, 950, 1200, 1375, 525, 700, 700, 500}; // Channels 12-19 (ADC)
const double TOP_SIPM_THRESHOLD = 450; // Channels 20-21 (ADC)

const double MICHEL_ENERGY_MIN = 40;    // Min PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX = 1000;  // Max PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX_DT = 400; // Max PMT energy for dt plots (p.e.)
const double MICHEL_DT_MIN = 0.8;       // Min time after muon for Michel (µs)
const double MICHEL_DT_MAX = 16.0;      // Max time after muon for Michel (µs)
const int MICHEL_PMT_MIN = 8;           // Min PMT hits for Michel

const double NUE_ENERGY_MIN = 352;      // Min PMT energy for νₑ (p.e.)
const double NUE_ENERGY_MAX = 1088;     // Max PMT energy for νₑ (p.e.)
const double NUE_VETO_THRESHOLD = 80;   // Max SiPM sum for νₑ (ADC)
const int NUE_PMT_MIN = 4;              // Min PMT hits for νₑ
const double NUE_DT_MIN = 1.0;          // Min time after beam for νₑ (µs)
const double NUE_DT_MAX = 10.0;         // Max time after beam for νₑ (µs)

const double COSMIC_SIPM_THRESHOLD = 500.0; // SiPM threshold for cosmic events (ADC)
const double UNTAGGED_SIPM_THRESHOLD = 150.0; // SiPM threshold for untagged events (ADC)
const double UNTAGGED_ENERGY_MAX = 80;  // Max PMT energy for untagged (p.e.)
const int UNTAGGED_PMT_MIN = 1;         // Min PMT hits for untagged
const int UNTAGGED_PMT_MAX = 12;         // Max PMT hits for untagged

// Pulse structure
struct myPulse {
    double start;          // Start time (µs)
    double end;            // End time (µs)
    double peak;           // Max amplitude (p.e. for PMTs, ADC for SiPMs)
    double energy;         // Energy (p.e. for PMTs, ADC for SiPMs)
    double number;         // Number of channels with pulse
    bool single;           // Timing consistency
    bool beam;             // Beam status
    int trigger;           // Trigger type
    double side_sipm_energy; // Side SiPM energy (ADC)
    double top_sipm_energy;  // Top SiPM energy (ADC)
    double all_sipm_energy;  // All SiPM energy (ADC)
    double last_mu;        // Time of last muon (µs)
    bool is_muon;          // Muon candidate flag
    bool is_michel;        // Michel electron candidate flag
    bool is_nue;           // νₑ candidate flag
    std::vector<double> pmt_energies; // Per-PMT energies (p.e.)
};

// Temporary pulse structure
struct tempPulse {
    double start;  // Start time (µs)
    double end;    // End time (µs)
    double peak;   // Max amplitude
    double energy; // Energy
};

// SPE fitting function
Double_t SPEfit(Double_t *x, Double_t *par) {
    Double_t A0 = par[0], mu0 = par[1], sigma0 = par[2];
    Double_t A1 = par[3], mu1 = par[4], sigma1 = par[5];
    Double_t A2 = par[6], A3 = par[7];
    Double_t term1 = A0 * exp(-0.5 * pow((x[0] - mu0) / sigma0, 2));
    Double_t term2 = A1 * exp(-0.5 * pow((x[0] - mu1) / sigma1, 2));
    Double_t term3 = A2 * exp(-0.5 * pow((x[0] - sqrt(2) * mu1) / sqrt(2 * sigma1 * sigma1 - sigma0 * sigma0), 2));
    Double_t term4 = A3 * exp(-0.5 * pow((x[0] - sqrt(3) * mu1) / sqrt(3 * sigma1 * sigma1 - 2 * sigma0 * sigma0), 2));
    return term1 + term2 + term3 + term4;
}

// Exponential fit function: N0 * exp(-t/tau) + C (t, tau in µs)
Double_t ExpFit(Double_t *exp_t, Double_t *t) {
    return t[0] * exp(-exp_t[0] / t[1]) + t[2];
}

// Utility functions
template<typename T>
T getAverage(const std::vector<T>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template<typename T>
double mostFrequent(const std::vector<T>& v) {
    if (v.empty()) return 0;
    std::map<T, double> most_frequent;
    for (const auto& val : v) most_frequent[val]++;
    T most_f = v[0];
    int max_count = 0;
    for (const auto& pair : most_frequent) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_f = pair.first;
        }
    }
    return max_count > 1 ? most_f : most_f;
}

template<typename T>
T variance(const std::vector<T>& v) {
    if (v.size() <= 1) return 0;
    T mean = getAverage(v);
    T sum = 0;
    for (const auto& val : v) {
        sum += (val - mean) * (val - mean);
    }
    return sum / (v.size() - 1);
}

// Create output directory
void makeOutputDir(const string& dirName) {
    struct stat st;
    if (stat(dirName.c_str(), &st)) {
        if (mkdir(dirName.c_str(), 0755)) {
            cout << "Error: Could not create directory " << dirName << endl;
            exit(1);
        }
        cout << "Created output directory: " << dirName << endl;
    } else {
        cout << "Output directory already exists: " << dirName << endl;
    }
}

// Selection criteria functions
bool is_sipm_hit_muon(const std::vector<double>& sipm_energies) {
    for (size_t i = 0; i < SIDE_SIPM_THRESHOLDS.size(); i++) {
        if (sipm_energies[i] > SIDE_SIPM_THRESHOLDS[i]) {
            return true;
        }
    }
    return sipm_energies[8] > TOP_SIPM_THRESHOLD || sipm_energies[9] > TOP_SIPM_THRESHOLD;
}

bool is_sipm_low(const std::vector<double>& sipm_energies) {
    for (size_t i = 0; i < SIDE_SIPM_THRESHOLDS.size(); i++) {
        if (sipm_energies[i] > SIDE_SIPM_THRESHOLDS[i]) {
            return false;
        }
    }
    return !(sipm_energies[8] > TOP_SIPM_THRESHOLD || sipm_energies[9] > TOP_SIPM_THRESHOLD);
}

bool is_sipm_hit_cosmic(const std::vector<double>& sipm_energies) {
    for (size_t i = 0; i < sipm_energies.size(); i++) {
        if (sipm_energies[i] > COSMIC_SIPM_THRESHOLD) {
            return true;
        }
    }
    return false;
}

bool is_sipm_hit_untagged(const std::vector<double>& sipm_energies) {
    for (size_t i = 0; i < sipm_energies.size(); i++) {
        if (sipm_energies[i] > UNTAGGED_SIPM_THRESHOLD) {
            return true;
        }
    }
    return false;
}

bool is_muon_event(const myPulse& p, const std::vector<double>& sipm_energies) {
    return p.energy > MUON_ENERGY_THRESHOLD && is_sipm_hit_muon(sipm_energies);
}

bool is_michel_event(const myPulse& p, const std::vector<double>& sipm_energies, double dt) {
    return p.energy >= MICHEL_ENERGY_MIN &&
           p.energy <= MICHEL_ENERGY_MAX &&
           dt >= MICHEL_DT_MIN &&
           dt <= MICHEL_DT_MAX &&
           p.number >= MICHEL_PMT_MIN &&
           is_sipm_low(sipm_energies) &&
           !(p.trigger == 1 || p.trigger == 4 || p.trigger == 8 || p.trigger == 16);
}

bool is_nue_event(const myPulse& p, double beam_dt) {
    return p.energy >= NUE_ENERGY_MIN &&
           p.energy <= NUE_ENERGY_MAX &&
           beam_dt >= NUE_DT_MIN &&
           beam_dt <= NUE_DT_MAX &&
           p.number >= NUE_PMT_MIN &&
           p.all_sipm_energy < NUE_VETO_THRESHOLD &&
           !p.is_muon &&
           !p.is_michel;
}

bool is_cosmic_event(const myPulse& p, const std::vector<double>& sipm_energies) {
    return (p.is_muon || p.is_michel || is_sipm_hit_cosmic(sipm_energies)) && p.energy > 1;
}

bool is_tagged_event(const myPulse& p, const std::vector<double>& sipm_energies) {
    return (is_sipm_hit_cosmic(sipm_energies) || p.is_muon || p.is_michel) && p.energy > 1;
}

bool is_untagged_event(const myPulse& p, const std::vector<double>& sipm_energies, double beam_dt, bool pulse_at_end) {
    bool is_non_muon_low_energy = p.energy <= UNTAGGED_ENERGY_MAX && !pulse_at_end && !is_sipm_hit_untagged(sipm_energies);
    return p.trigger == 2 &&
           !is_sipm_hit_untagged(sipm_energies) &&
           p.number >= UNTAGGED_PMT_MIN &&
           p.number <= UNTAGGED_PMT_MAX &&
           is_non_muon_low_energy &&
           !p.beam;
}

bool is_veto_passing_event(const myPulse& p, const std::vector<double>& sipm_energies) {
    return !p.is_muon &&
           !p.is_michel &&
           is_sipm_low(sipm_energies) &&
           p.number >= 1 &&
           p.energy > 1 &&
           p.trigger == 2;
}

// SPE calibration function
void myCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cout << "Error opening calibration file: " << calibFileName << endl;
        exit(1);
    }
    TTree *calib = (TTree*)calibFile->Get("tree");
    if (!calib) {
        cout << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        delete calibFile;
        exit(1);
    }
    Short_t adcVal[23][45];
    Double_t area[23];
    Int_t triggerBits;
    calib->SetBranchAddress("adcVal", adcVal);
    calib->SetBranchAddress("area", area);
    calib->SetBranchAddress("triggerBits", &triggerBits);
    Long64_t nEntries = calib->GetEntries();
    cout << "Processing " << nEntries << " calibration events from " << calibFileName << "..." << endl;
    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i + 1),
                               Form("PMT %d;Area;Events per 3 ADCs", i + 1),
                               150, -50, 400);
        histArea[i]->SetLineColor(kRed);
        histArea[i]->GetXaxis()->SetLabelFont(42);
        histArea[i]->GetYaxis()->SetLabelFont(42);
        histArea[i]->GetXaxis()->SetTitleFont(42);
        histArea[i]->GetYaxis()->SetTitleFont(42);
    }
    for (Long64_t ev = 0; ev < nEntries; ++ev) {
        calib->GetEntry(ev);
        if (triggerBits == 16) {
            for (int p = 0; p < N_PMTS; ++p) {
                histArea[p]->Fill(area[PMT_CHANNEL_MAP[p]]);
                nLEDFlashes[p]++;
            }
        }
    }
    TCanvas *canvas = new TCanvas("canvas", "PMT Energy Distributions", 1200, 800);
    canvas->SetLeftMargin(0.15);
    canvas->SetRightMargin(0.05);
    canvas->SetBottomMargin(0.15);
    canvas->SetTopMargin(0.05);
    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cout << "Warning: Insufficient data for PMT " << i + 1 << " in " << calibFileName << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            continue;
        }
        canvas->Clear();
        histArea[i]->GetXaxis()->SetTitleSize(0.05);
        histArea[i]->GetYaxis()->SetTitleSize(0.05);
        histArea[i]->GetXaxis()->SetLabelSize(0.04);
        histArea[i]->GetYaxis()->SetLabelSize(0.04);
        TF1 *f = new TF1("f", SPEfit, -50, 400, 8);
        f->SetParameters(1000, 0, 10, 1000, 50, 10, 500, 500);
        f->SetLineColor(kBlue);
        f->SetParNames("A0", "#mu_{0}", "#sigma_{0}", "A1", "#mu_{1}", "#sigma_{1}", "A2", "A3");
        histArea[i]->Fit(f, "R");
        histArea[i]->Draw();
        f->Draw("same");
        mu1[i] = f->GetParameter(4);
        mu1_err[i] = f->GetParError(4);
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->SetTextSize(0.06);
        tex->SetTextAlign(22);
        tex->SetNDC();
        tex->DrawLatex(0.5, 0.92, Form("PMT %d", i + 1));
        gPad->Update();
        if (auto stats = (TPaveStats*)histArea[i]->FindObject("stats")) {
            stats->SetX1NDC(0.65);
            stats->SetY1NDC(0.65);
            stats->SetX2NDC(0.95);
            stats->SetY2NDC(0.95);
            stats->SetTextFont(42);
            stats->SetTextSize(0.03);
            stats->SetOptStat(10);
            stats->SetOptFit(111);
            stats->SetName("");
        }
        canvas->SaveAs(Form("%s/PMT%d_Energy_Distribution.png", OUTPUT_DIR.c_str(), i + 1));
        delete f;
        delete tex;
    }
    delete canvas;
    gStyle->SetTextFont(42);
    gStyle->SetLabelFont(42, "XYZ");
    gStyle->SetTitleFont(42, "XYZ");
    TCanvas *master = new TCanvas("MasterCanvas", "Combined PMT Energy Distributions", 1200, 800);
    master->Divide(3, 4, 0, 0);
    int layout[4][3] = {
        {0, 10, 7},
        {2, 6, 3},
        {8, 9, 11},
        {4, 5, 1}
    };
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            int pad = r * 3 + c + 1;
            master->cd(pad);
            int idx = layout[r][c];
            histArea[idx]->GetXaxis()->SetTitleSize(0.06);
            histArea[idx]->GetYaxis()->SetTitleSize(0.08);
            histArea[idx]->GetXaxis()->SetLabelSize(0.04);
            histArea[idx]->GetYaxis()->SetLabelSize(0.04);
            histArea[idx]->GetYaxis()->SetTitle("Events per 3 ADCs");
            histArea[idx]->GetYaxis()->SetTitleOffset(0.8);
            histArea[idx]->GetXaxis()->SetTitle("Area");
            gPad->SetLeftMargin(0.15);
            gPad->SetRightMargin(0.12);
            gPad->SetBottomMargin(0.15);
            gPad->SetTopMargin(0.10);
            TF1 *f2 = new TF1("f2", SPEfit, -50, 400, 8);
            f2->SetParameters(100, 0, 10, 100, 50, 10, 50, 50);
            f2->SetLineColor(kBlue);
            f2->SetParNames("A0", "#mu_{0}", "#sigma_{0}", "A1", "#mu_{1}", "#sigma_{1}", "A2", "A3");
            histArea[idx]->Fit(f2, "R");
            histArea[idx]->Draw();
            f2->Draw("same");
            TLatex *tex2 = new TLatex();
            tex2->SetTextFont(42);
            tex2->SetTextSize(0.08);
            tex2->SetTextAlign(22);
            tex2->SetNDC();
            tex2->DrawLatex(0.5, 0.92, Form("PMT %d", idx + 1));
            gPad->Update();
            if (auto s2 = (TPaveStats*)histArea[idx]->FindObject("stats")) {
                s2->SetX1NDC(0.65);
                s2->SetY1NDC(0.65);
                s2->SetX2NDC(0.95);
                s2->SetY2NDC(0.95);
                s2->SetTextFont(42);
                s2->SetTextSize(0.02);
                s2->SetOptStat(10);
                s2->SetOptFit(111);
                s2->SetName("");
            }
            delete f2;
            delete tex2;
        }
    }
    master->SaveAs(Form("%s/Combined_PMT_Energy_Distributions.pdf", OUTPUT_DIR.c_str()));
    master->SaveAs(Form("%s/Combined_PMT_Energy_Distributions.png", OUTPUT_DIR.c_str()));
    for (int i = 0; i < N_PMTS; ++i) delete histArea[i];
    delete master;
    calibFile->Close();
    delete calibFile;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <calibration_file> <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }
    string calibFileName = argv[1];
    vector<string> inputFiles;
    for (int i = 2; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }
    makeOutputDir(OUTPUT_DIR);
    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }
    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cout << "Error: Calibration file " << calibFileName << " not found" << endl;
        return -1;
    }
    bool anyInputFileExists = false;
    for (const auto& file : inputFiles) {
        if (!gSystem->AccessPathName(file.c_str())) {
            anyInputFileExists = true;
            break;
        }
    }
    if (!anyInputFileExists) {
        cout << "Error: No input files found" << endl;
        return -1;
    }
    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    myCalibration(calibFileName, mu1, mu1_err);
    cout << "SPE Calibration Results (from " << calibFileName << "):\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }
    int num_events = 0;
    int num_muons = 0;
    int num_michels = 0;
    int num_nue = 0;
    int total_num_untagged = 0;
    int total_num_veto_pass = 0;
    int cosmic_veto_pass = 0, cosmic_veto_fail = 0;
    int tagged_veto_pass = 0, tagged_veto_fail = 0;
    int untagged_veto_pass = 0;
    std::map<int, int> trigger_counts;
    std::vector<double> beam_pulse_times;
    TH1D* hMyMuonEnergy = new TH1D("muon_energy", "Muon Energy Spectrum;Energy (p.e.);Counts/100 p.e.", 550, -500, 5000);
    TH1D* hMyMichelEnergy = new TH1D("michel_energy", "Michel Electron Energy;Energy (p.e.);Counts/4 p.e.", 200, 0, 800);
    TH1D* hMyDtMichel = new TH1D("DeltaT", "Muon-Michel Time Difference;Time (#mus);Counts/0.1 #mus", 160, 0, MICHEL_DT_MAX);
    TH2D* hMyEnergyVsDt = new TH2D("energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, MICHEL_DT_MAX, 100, 0, 1000);
    TH1D* hMyTriggerBits = new TH1D("trigger_bits", "Trigger Bits Distribution;Trigger Bits;Counts", 36, 0, 36);
    TH1D* hMyNueEnergy = new TH1D("nue_energy", "νₑ Energy Distribution;Energy (p.e.);Counts/10 p.e.", 150, 0, 1500);
    TH1D* hMyNueDt = new TH1D("nue_dt", "νₑ Time Difference;Time to Beam (#mus);Counts/Events", 120, 0, 12);
    TH1D* hMyNuePmtHits = new TH1D("nue_pmt_hits", "νₑ PMT Hits;Number of PMTs;Counts", 12, 0, 12);
    TH2D* hMyNueEnergyVsDt = new TH2D("nue_energy_vs_dt", "νₑ Energy vs Time Difference;dt (#mus);Energy (p.e.)", 120, 0, 12, 150, 0, 1500);
    TH1D* hMyNueBkgEnergy = new TH1D("nue_bkg_energy", "νₑ Background Energy;Energy (p.e.);Counts/10 p.e.", 150, 0, 1500);
    TH1D* h_pmt_energy_all = new TH1D("pmt_energy_all", "PMT Energy for All Events;Energy (p.e.);Events", 100, 0, 1500);
    TH1D* h_pmt_energy_after_veto = new TH1D("pmt_energy_after_veto", "PMT Energy After Veto;Energy (p.e.);Events", 100, 0, 1500);
    TH1D* h_pmt_energy_cosmic = new TH1D("pmt_energy_cosmic", "PMT Energy for Cosmic Events;Energy (p.e.);Events", 100, 0, 1500);
    TH1D* h_pmt_energy_untagged = new TH1D("pmt_energy_untagged", "PMT Energy for Untagged Events;Energy (p.e.);Events", 100, 0, 1500);
    TH1D* h_pmt_energy_tagged = new TH1D("pmt_energy_tagged", "PMT Energy for Tagged Events;Energy (p.e.);Events", 100, 0, 1500);
    TH1D* h_pmt_energy_veto_pass = new TH1D("pmt_energy_veto_pass", "PMT Energy for Veto-Passing Events;Energy (p.e.);Events", 100, 0, 1500);
    TH1D* h_sipm_untagged = new TH1D("sipm_untagged", "SiPM Energy for Untagged Events;SiPM Energy (ADC);Events", 100, 0, 200);
    TH1D* h_pmt_hits_untagged = new TH1D("pmt_hits_untagged", "PMT Hits for Untagged Events;Number of PMTs;Events", 12, 0, 12);
    TH1D* h_dt_untagged = new TH1D("dt_untagged", "Time to Beam for Untagged Events;dt (#mus);Events", 120, 0, 12);
    TH1D* h_sipm_veto_pass = new TH1D("sipm_veto_pass", "SiPM Energy for Veto-Passing;SiPM Energy (ADC);Events", 100, 0, 100);
    TH1D* h_dt_all = new TH1D("dt_all", "Time to Last Muon;dt (#mus);Events", 160, 0, 16);
    TH1D* h_after_veto_trigger2 = new TH1D("after_veto_trigger2", "After Veto (Trigger 2);Energy (p.e.);Events", 100, 0, 1500);
    TH1D* h_sipm_cosmic = new TH1D("sipm_cosmic", "SiPM Energy for Cosmic Events;SiPM Energy (ADC);Events", 100, 0, 500);
    TH1D* h_sipm_tagged = new TH1D("sipm_tagged", "SiPM Energy for Tagged Events;SiPM Energy (ADC);Events", 100, 0, 500);
    TH1D* h_pmt_energy_all_pmt[N_PMTS];
    TH1D* h_pmt_energy_veto_pass_pmt[N_PMTS];
    for (int i = 0; i < N_PMTS; ++i) {
        h_pmt_energy_all_pmt[i] = new TH1D(Form("pmt_energy_all_pmt%d", i + 1),
                                           Form("PMT %d Energy for All Events;Energy (p.e.);Events", i + 1),
                                           100, 0, 1000);
        h_pmt_energy_veto_pass_pmt[i] = new TH1D(Form("pmt_energy_veto_pass_pmt%d", i + 1),
                                                Form("PMT %d Energy for Veto-Passing Events;Energy (p.e.);Events", i + 1),
                                                100, 0, 1000);
    }

    for (const auto& inputFileName : inputFiles) {
        TFile *f = TFile::Open(inputFileName.c_str());
        if (!f || f->IsZombie()) {
            cout << "Error: Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }
        cout << "Processing file: " << inputFileName << endl;
        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Error: Could not find tree in file: " << inputFileName << endl;
            f->Close();
            delete f;
            continue;
        }
        Int_t eventID;
        Int_t nSamples[23];
        Short_t adcVal[23][45];
        Double_t baselineMean[23];
        Double_t baselineRMS[23];
        Double_t pulseH[23];
        Int_t peakPosition[23];
        Double_t area[23];
        Long64_t nsTime;
        Int_t triggerBits;
        t->SetBranchAddress("eventID", &eventID);
        t->SetBranchAddress("nSamples", nSamples);
        t->SetBranchAddress("adcVal", adcVal);
        t->SetBranchAddress("baselineMean", baselineMean);
        t->SetBranchAddress("baselineRMS", baselineRMS);
        t->SetBranchAddress("pulseH", pulseH);
        t->SetBranchAddress("peakPosition", peakPosition);
        t->SetBranchAddress("area", area);
        t->SetBranchAddress("nsTime", &nsTime);
        t->SetBranchAddress("triggerBits", &triggerBits);
        int numEntries = t->GetEntries();
        cout << "Collecting beam pulses from " << numEntries << " entries in " << inputFileName << endl;
        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            if (triggerBits & (1 | 2 | 3)) {
                double evbf_energy = 0;
                for (int i = 0; i < ADCSIZE; i++) {
                    evbf_energy += adcVal[22][i] - baselineMean[22];
                }
                if (evbf_energy > EVBF_THRESHOLD) {
                    beam_pulse_times.push_back(nsTime / 1000.0 + BEAM_TIME_CORRECTION);
                }
            }
        }
        double last_muon_time = 0.0;
        std::set<double> michel_muon_times;
        std::vector<std::pair<double, double>> muon_candidates;
        cout << "Processing " << numEntries << " entries for event detection in " << inputFileName << endl;
        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;
            hMyTriggerBits->Fill(triggerBits);
            trigger_counts[triggerBits]++;
            if (triggerBits < 0 || triggerBits > 36) {
                cout << "Warning: triggerBits = " << triggerBits << " out of histogram range (0-36) in file " << inputFileName << ", event " << eventID << endl;
            }
            struct myPulse p;
            p.start = nsTime / 1000.0;
            p.end = nsTime / 1000.0;
            p.peak = 0;
            p.energy = 0;
            p.number = 0;
            p.single = false;
            p.beam = false;
            p.trigger = triggerBits;
            p.side_sipm_energy = 0;
            p.top_sipm_energy = 0;
            p.all_sipm_energy = 0;
            p.last_mu = last_muon_time;
            p.is_muon = false;
            p.is_michel = false;
            p.is_nue = false;
            p.pmt_energies = std::vector<double>(N_PMTS, 0.0);
            std::vector<double> all_chan_start, all_chan_end, all_chan_peak, all_chan_energy;
            std::vector<double> side_sipm_energy, top_sipm_energy;
            std::vector<double> chan_starts_no_outliers;
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);
            bool pulse_at_end = false;
            std::vector<double> sipm_energies(10, 0);
            for (int iChan = 0; iChan < 23; iChan++) {
                int pulse_at_end_count = 0;
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }
                if (iChan == 22) {
                    double evbf_energy = 0;
                    for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                        evbf_energy += h_wf.GetBinContent(iBin);
                    }
                    if (evbf_energy > EVBF_THRESHOLD) {
                        p.beam = true;
                    }
                }
                std::vector<tempPulse> pulses_temp;
                bool onPulse = false;
                int thresholdBin = 0, peakBin = 0;
                double peak = 0, pulseEnergy = 0;
                double allPulseEnergy = 0;
                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    double iBinContent = h_wf.GetBinContent(iBin);
                    if (iBin > 15) allPulseEnergy += iBinContent;
                    if (!onPulse && iBinContent >= PULSE_THRESHOLD) {
                        onPulse = true;
                        thresholdBin = iBin;
                        peakBin = iBin;
                        peak = iBinContent;
                        pulseEnergy = iBinContent;
                    } else if (onPulse) {
                        pulseEnergy += iBinContent;
                        if (peak < iBinContent) {
                            peak = iBinContent;
                            peakBin = iBin;
                        }
                        if (iBinContent < PULSE_THRESHOLD || iBin == ADCSIZE) {
                            tempPulse pt;
                            pt.start = thresholdBin * 16.0 / 1000.0;
                            pt.peak = iChan <= 11 && mu1[iChan] > 1 ? peak / mu1[iChan] : peak;
                            pt.end = iBin * 16.0 / 1000.0;
                            for (int j = peakBin - 1; j >= 1; j--) {
                                if (h_wf.GetBinContent(j) > peak * 0.1) {
                                    pt.start = j * 16.0 / 1000.0;
                                }
                                pulseEnergy += h_wf.GetBinContent(j);
                            }
                            if (iChan <= 11 && mu1[iChan] > 1) {
                                pt.energy = pulseEnergy / mu1[iChan];
                                all_chan_start.push_back(pt.start);
                                all_chan_end.push_back(pt.end);
                                all_chan_peak.push_back(pt.peak);
                                all_chan_energy.push_back(pt.energy);
                                if (pt.energy > 1) p.number += 1;
                                for (int pmt = 0; pmt < N_PMTS; ++pmt) {
                                    if (PMT_CHANNEL_MAP[pmt] == iChan) {
                                        p.pmt_energies[pmt] = pt.energy;
                                        break;
                                    }
                                }
                            }
                            pulses_temp.push_back(pt);
                            peak = 0;
                            peakBin = 0;
                            pulseEnergy = 0;
                            thresholdBin = 0;
                            onPulse = false;
                        }
                    }
                }
                if (iChan >= 12 && iChan <= 21) {
                    sipm_energies[iChan - 12] = allPulseEnergy;
                    if (iChan <= 19) side_sipm_energy.push_back(allPulseEnergy);
                    else top_sipm_energy.push_back(allPulseEnergy);
                }
                if (iChan <= 11 && h_wf.GetBinContent(ADCSIZE) > 100) {
                    pulse_at_end_count++;
                    if (pulse_at_end_count >= 10) pulse_at_end = true;
                }
                h_wf.Reset();
            }
            p.start += mostFrequent(all_chan_start);
            p.end += mostFrequent(all_chan_end);
            p.energy = std::accumulate(all_chan_energy.begin(), all_chan_energy.end(), 0.0);
            p.peak = std::accumulate(all_chan_peak.begin(), all_chan_peak.end(), 0.0);
            p.side_sipm_energy = std::accumulate(side_sipm_energy.begin(), side_sipm_energy.end(), 0.0);
            p.top_sipm_energy = std::accumulate(top_sipm_energy.begin(), top_sipm_energy.end(), 0.0);
            p.all_sipm_energy = p.side_sipm_energy + p.top_sipm_energy;
            for (const auto& start : all_chan_start) {
                if (fabs(start - mostFrequent(all_chan_start)) < 10 * 16.0 / 1000.0) {
                    chan_starts_no_outliers.push_back(start);
                }
            }
            p.single = (variance(chan_starts_no_outliers) < 5 * 16.0 / 1000.0);
            double dt = p.start - last_muon_time;
            bool no_event61 = true;
            double evbf_energy = 0;
            for (int i = 0; i < ADCSIZE; i++) {
                evbf_energy += adcVal[22][i] - baselineMean[22];
            }
            if (evbf_energy > EVBF_THRESHOLD) {
                no_event61 = false;
            }
            // Muon selection
            if (is_muon_event(p, sipm_energies)) {
                p.is_muon = true;
                last_muon_time = p.start;
                num_muons++;
                muon_candidates.emplace_back(p.start, p.energy);
            }
            // Michel selection
            bool is_michel_candidate = is_michel_event(p, sipm_energies, dt);
            bool is_michel_for_dt = is_michel_candidate && p.energy <= MICHEL_ENERGY_MAX_DT;
            if (is_michel_candidate) {
                p.is_michel = true;
                num_michels++;
                michel_muon_times.insert(last_muon_time);
                hMyMichelEnergy->Fill(p.energy);
            }
            if (is_michel_for_dt) {
                hMyDtMichel->Fill(dt);
                hMyEnergyVsDt->Fill(dt, p.energy);
            }
            // νₑ selection
            double beam_dt = 1e9;
            bool is_beam_triggered = p.trigger & (1 | 2 | 3);
            if (is_beam_triggered) {
                for (const auto& beam_time : beam_pulse_times) {
                    double dt_beam = p.start - beam_time;
                    if (dt_beam > 0 && dt_beam < beam_dt) {
                        beam_dt = dt_beam;
                    }
                }
                if (is_nue_event(p, beam_dt)) {
                    p.is_nue = true;
                    num_nue++;
                    hMyNueEnergy->Fill(p.energy);
                    hMyNueDt->Fill(beam_dt);
                    hMyNuePmtHits->Fill(p.number);
                    hMyNueEnergyVsDt->Fill(beam_dt, p.energy);
                } else {
                    hMyNueBkgEnergy->Fill(p.energy);
                }
            }
            // All events
            bool is_valid_trigger = !(p.trigger == 4 || p.trigger == 8 || p.trigger == 16);
            if (is_valid_trigger) {
                h_pmt_energy_all->Fill(p.energy);
                for (int pmt = 0; pmt < N_PMTS; ++pmt) {
                    if (p.pmt_energies[pmt] > 0) {
                        h_pmt_energy_all_pmt[pmt]->Fill(p.pmt_energies[pmt]);
                    }
                }
            }
            // Cosmic selection
            if (is_cosmic_event(p, sipm_energies)) {
                h_pmt_energy_cosmic->Fill(p.energy);
                h_sipm_cosmic->Fill(p.all_sipm_energy);
                if (is_veto_passing_event(p, sipm_energies)) cosmic_veto_pass++;
                else cosmic_veto_fail++;
            }
            // Tagged selection
            if (is_tagged_event(p, sipm_energies)) {
                h_pmt_energy_tagged->Fill(p.energy);
                h_sipm_tagged->Fill(p.all_sipm_energy);
                if (is_veto_passing_event(p, sipm_energies)) tagged_veto_pass++;
                else tagged_veto_fail++;
            }
            // Untagged selection
            if (is_untagged_event(p, sipm_energies, beam_dt, pulse_at_end)) {
                h_pmt_energy_untagged->Fill(p.energy);
                h_sipm_untagged->Fill(p.all_sipm_energy);
                h_pmt_hits_untagged->Fill(p.number);
                h_dt_untagged->Fill(beam_dt);
                total_num_untagged++;
                if (is_veto_passing_event(p, sipm_energies)) untagged_veto_pass++;
            }
            // Veto-passing selection
            if (is_veto_passing_event(p, sipm_energies)) {
                h_pmt_energy_veto_pass->Fill(p.energy);
                h_pmt_energy_after_veto->Fill(p.energy);
                h_sipm_veto_pass->Fill(p.all_sipm_energy);
                h_after_veto_trigger2->Fill(p.energy);
                for (int pmt = 0; pmt < N_PMTS; ++pmt) {
                    if (p.pmt_energies[pmt] > 0) {
                        h_pmt_energy_veto_pass_pmt[pmt]->Fill(p.pmt_energies[pmt]);
                    }
                }
                total_num_veto_pass++;
            }
            // Time to muon
            if (p.start > last_muon_time) {
                h_dt_all->Fill(p.start - last_muon_time);
            }
            p.last_mu = last_muon_time;
        }
        for (const auto& muon : muon_candidates) {
            if (michel_muon_times.find(muon.first) != michel_muon_times.end()) {
                hMyMuonEnergy->Fill(muon.second);
            }
        }
        cout << "File " << inputFileName << " Statistics:\n";
        cout << "Total Events: " << num_events << "\n";
        cout << "Muons Detected: " << num_muons << "\n";
        cout << "Michel Electrons Detected: " << num_michels << "\n";
        cout << "νₑ Events Detected: " << num_nue << "\n";
        cout << "Untagged Events: " << total_num_untagged << "\n";
        cout << "Veto-Passing Events: " << total_num_veto_pass << "\n";
        cout << "------------------------\n";
        f->Close();
        delete f;
        num_events = 0;
        num_muons = 0;
        num_michels = 0;
        num_nue = 0;
    }
    cout << "Trigger Bits Distribution (all files):\n";
    for (const auto& pair : trigger_counts) {
        cout << "Trigger " << pair.first << ": " << pair.second << " events\n";
    }
    cout << "------------------------\n";
    cout << "Veto Efficiency Analysis:\n";
    cout << "Cosmic Rejection Efficiency: " << (cosmic_veto_fail / (double)(cosmic_veto_pass + cosmic_veto_fail)) << " (" << cosmic_veto_fail << "/" << (cosmic_veto_pass + cosmic_veto_fail) << ")\n";
    cout << "Tagged Rejection Efficiency: " << (tagged_veto_fail / (double)(tagged_veto_pass + tagged_veto_fail)) << " (" << tagged_veto_fail << "/" << (tagged_veto_pass + tagged_veto_fail) << ")\n";
    cout << "Untagged Retention Efficiency: " << (untagged_veto_pass / (double)total_num_untagged) << " (" << untagged_veto_pass << "/" << total_num_untagged << ")\n";
    cout << "------------------------\n";
    TCanvas *c = new TCanvas("c", "Analysis Plots", 1200, 800);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(111);
    c->Clear();
    hMyMuonEnergy->SetLineColor(kBlue);
    hMyMuonEnergy->Draw();
    c->Update();
    string plotName = OUTPUT_DIR + "/Muon_Energy_Michel.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyMichelEnergy->SetLineColor(kRed);
    hMyMichelEnergy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyDtMichel->SetMarkerStyle(20);
    hMyDtMichel->SetMarkerSize(1.0);
    hMyDtMichel->GetXaxis()->SetTitle("Time to Previous Muon (#mus)");
    hMyDtMichel->Draw("P");
    if (hMyDtMichel->GetEntries() > 5) {
        double integral = hMyDtMichel->Integral(hMyDtMichel->FindBin(FIT_MIN), hMyDtMichel->FindBin(FIT_MAX));
        double bin_width = hMyDtMichel->GetBinWidth(1);
        double N0_init = integral * bin_width / (FIT_MAX - FIT_MIN);
        double C_init = 0;
        int bin_14 = hMyDtMichel->FindBin(14.0);
        int bin_16 = hMyDtMichel->FindBin(16.0);
        double min_content = 1e9;
        for (int i = bin_14; i <= bin_16; i++) {
            double content = hMyDtMichel->GetBinContent(i);
            if (content > 0 && content < min_content) {
                min_content = content;
            }
        }
        if (min_content < 1e9) {
            C_init = min_content;
        } else {
            C_init = 0.1;
        }
        TF1 *expFit = new TF1("expFit", ExpFit, FIT_MIN, FIT_MAX, 3);
        expFit->SetParameters(N0_init, 2.2, C_init);
        expFit->SetParLimits(0, 0, N0_init * 100);
        expFit->SetParLimits(1, 0.1, 20.0);
        expFit->SetParLimits(2, -C_init * 10, C_init * 10);
        expFit->SetParNames("N_{0}", "#tau", "C");
        expFit->SetNpx(1000);
        int fitStatus = hMyDtMichel->Fit(expFit, "RE", "", FIT_MIN, FIT_MAX);
        expFit->SetLineColor(kGreen);
        expFit->SetLineWidth(3);
        expFit->Draw("same");
        gPad->Update();
        cout << "Fit line drawn with color kGreen, width 3" << endl;
        TPaveStats* stats = (TPaveStats*)hMyDtMichel->FindObject("stats");
        if (!stats) {
            cout << "Stats box not found, creating new TPaveStats" << endl;
            stats = new TPaveStats(0.6, 0.6, 0.9, 0.9, "brNDC");
            stats->SetName("stats");
            stats->Draw();
            hMyDtMichel->GetListOfFunctions()->Add(stats);
        } else {
            cout << "Stats box found, updating content" << endl;
        }
        stats->SetTextColor(kRed);
        stats->SetX1NDC(0.6);
        stats->SetX2NDC(0.9);
        stats->SetY1NDC(0.6);
        stats->SetY2NDC(0.9);
        stats->Clear();
        stats->AddText("DeltaT");
        stats->AddText(Form("#tau = %.4f #pm %.4f #mus", expFit->GetParameter(1), expFit->GetParError(1)));
        stats->AddText(Form("#chi^{2}/NDF = %.4f", expFit->GetChisquare() / expFit->GetNDF()));
        stats->AddText(Form("N_{0} = %.1f #pm %.1f", expFit->GetParameter(0), expFit->GetParError(0)));
        stats->AddText(Form("C = %.1f #pm %.1f", expFit->GetParameter(2), expFit->GetParError(2)));
        stats->Draw();
        gPad->Update();
        double N0 = expFit->GetParameter(0);
        double N0_err = expFit->GetParError(0);
        double tau = expFit->GetParameter(1);
        double tau_err = expFit->GetParError(1);
        double C = expFit->GetParameter(2);
        double C_err = expFit->GetParError(2);
        double chi2 = expFit->GetChisquare();
        int ndf = expFit->GetNDF();
        double chi2_ndf = chi2;
        if (ndf > 0) {
            chi2_ndf = chi2 / ndf;
        }
        cout << "Exponential Fit Results (Michel dt, 1.0-10.0 μs):\n";
        cout << Form("Fit Status: %d (0 = success)", fitStatus) << endl;
        cout << Form("N₀ = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("τ = %.4f ± %.4f μs", tau, tau_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("Chi² = %.1f", chi2) << endl;
        cout << Form("NDF = %d", ndf) << endl;
        cout << Form("Chi²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;
        if (fitStatus != 0) {
            cout << "Warning: Exponential fit failed for hMyDtMichel (status = " << fitStatus << ")" << endl;
            cout << "Initial Parameters: N0 = " << N0_init << ", τ = 2.2 μs, C = " << C_init << endl;
            cout << "Fit results may be unreliable, but drawn for verification." << endl;
        }
        delete expFit;
    } else {
        cout << "Warning: hMyDtMichel has insufficient entries (" << hMyDtMichel->GetEntries() << ") for exponential fit" << endl;
        cout << "Check Michel electron detection criteria (e.g., MICHEL_ENERGY_MIN, p.number, SiPM thresholds)." << endl;
    }
    c->Update();
    c->Modified();
    c->RedrawAxis();
    plotName = OUTPUT_DIR + "/Michel_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyEnergyVsDt->SetStats(0);
    hMyEnergyVsDt->GetXaxis()->SetTitle("dt (#mus)");
    hMyEnergyVsDt->GetXaxis()->SetRangeUser(0, MICHEL_DT_MAX);
    hMyEnergyVsDt->GetYaxis()->SetRangeUser(0, 1000);
    hMyEnergyVsDt->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyTriggerBits->SetLineColor(kGreen);
    hMyTriggerBits->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/TriggerBits_Distribution.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyNueEnergy->SetLineColor(kBlue);
    hMyNueEnergy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Nue_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyNueDt->SetLineColor(kBlue);
    hMyNueDt->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Nue_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyNuePmtHits->SetLineColor(kBlue);
    hMyNuePmtHits->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Nue_PMT_Hits.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyNueEnergyVsDt->SetStats(0);
    hMyNueEnergyVsDt->GetXaxis()->SetTitle("dt (#mus)");
    hMyNueEnergyVsDt->GetXaxis()->SetRangeUser(0, 12);
    hMyNueEnergyVsDt->GetYaxis()->SetRangeUser(0, 1500);
    hMyNueEnergyVsDt->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Nue_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    hMyNueBkgEnergy->SetLineColor(kMagenta);
    hMyNueBkgEnergy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Nue_Bkg_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    h_sipm_cosmic->SetLineColor(kGreen);
    h_sipm_cosmic->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/SiPM_Energy_Cosmic.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    c->Clear();
    h_sipm_tagged->SetLineColor(kRed);
    h_sipm_tagged->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/SiPM_Energy_Tagged.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    TCanvas* c_pmt_veto = new TCanvas("c_pmt_veto", "PMT Energy All vs After Veto", 1200, 800);
    h_pmt_energy_all->SetLineColor(kRed);
    h_pmt_energy_after_veto->SetLineColor(kBlue);
    h_pmt_energy_all->SetLineWidth(2);
    h_pmt_energy_after_veto->SetLineWidth(2);
    h_pmt_energy_all->SetTitle("PMT Energy All vs After Veto;Energy (p.e.);Events");
    double max_veto = std::max(h_pmt_energy_all->GetMaximum(), h_pmt_energy_after_veto->GetMaximum());
    h_pmt_energy_all->SetMaximum(max_veto * 1.2);
    h_pmt_energy_all->SetMinimum(0.1);
    h_pmt_energy_all->Draw();
    h_pmt_energy_after_veto->Draw("SAME");
    gPad->SetLogy(1);
    TLegend* leg_veto = new TLegend(0.4, 0.75, 0.6, 0.85);
    leg_veto->AddEntry(h_pmt_energy_all, "All Events", "l");
    leg_veto->AddEntry(h_pmt_energy_after_veto, "After Veto", "l");
    leg_veto->Draw();
    c_pmt_veto->Update();
    plotName = OUTPUT_DIR + "/PMT_Energy_All_vs_AfterVeto.png";
    c_pmt_veto->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    TCanvas* c_pmt_categories = new TCanvas("c_pmt_categories", "PMT Energy by Category", 1200, 800);
    h_pmt_energy_cosmic->SetLineColor(kGreen);
    h_pmt_energy_untagged->SetLineColor(kBlack);
    h_pmt_energy_tagged->SetLineColor(kRed);
    h_pmt_energy_veto_pass->SetLineColor(kBlue);
    h_pmt_energy_cosmic->SetLineWidth(4);
    h_pmt_energy_untagged->SetLineWidth(3);
    h_pmt_energy_tagged->SetLineWidth(2);
    h_pmt_energy_veto_pass->SetLineWidth(2);
    h_pmt_energy_untagged->SetLineStyle(2);
    h_pmt_energy_tagged->SetLineStyle(7);
    h_pmt_energy_cosmic->SetLineStyle(1);
    h_pmt_energy_cosmic->SetTitle("PMT Energy by Category;Energy (p.e.);Events");
    cout << "Histogram Entries Before Plotting:\n";
    cout << "Cosmic Events: " << h_pmt_energy_cosmic->GetEntries() << endl;
    cout << "Tagged Events: " << h_pmt_energy_tagged->GetEntries() << endl;
    cout << "Untagged Events: " << h_pmt_energy_untagged->GetEntries() << endl;
    cout << "Veto-Passing Events: " << h_pmt_energy_veto_pass->GetEntries() << endl;
    double max_events_categories = std::max({h_pmt_energy_cosmic->GetMaximum(),
                                            h_pmt_energy_untagged->GetMaximum(),
                                            h_pmt_energy_tagged->GetMaximum(),
                                            h_pmt_energy_veto_pass->GetMaximum()});
    h_pmt_energy_cosmic->SetMaximum(max_events_categories * 2.0);
    h_pmt_energy_cosmic->SetMinimum(0.1);
    h_pmt_energy_cosmic->Draw("HIST");
    h_pmt_energy_tagged->Draw("SAME HIST");
    h_pmt_energy_untagged->Draw("SAME HIST");
    h_pmt_energy_veto_pass->Draw("SAME HIST");
    gPad->SetLogy();
    TLegend* leg_categories = new TLegend(0.4, 0.75, 0.6, 0.85);
    leg_categories->AddEntry(h_pmt_energy_cosmic, "Cosmic Events", "l");
    leg_categories->AddEntry(h_pmt_energy_tagged, "Tagged Events", "l");
    leg_categories->AddEntry(h_pmt_energy_untagged, "Untagged Events", "l");
    leg_categories->AddEntry(h_pmt_energy_veto_pass, "Passing veto Cut", "l");
    leg_categories->Draw();
    c_pmt_categories->Update();
    plotName = OUTPUT_DIR + "/PMT_Energy_Categories.png";
    c_pmt_categories->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    TCanvas* c_pmt_individual = new TCanvas("c_pmt_individual", "PMT Individual Energy", 1200, 800);
    for (int pmt = 0; pmt < N_PMTS; ++pmt) {
        c_pmt_individual->Clear();
        h_pmt_energy_all_pmt[pmt]->SetLineColor(kRed);
        h_pmt_energy_veto_pass_pmt[pmt]->SetLineColor(kBlue);
        h_pmt_energy_all_pmt[pmt]->SetLineWidth(2);
        h_pmt_energy_veto_pass_pmt[pmt]->SetLineWidth(2);
        h_pmt_energy_all_pmt[pmt]->SetTitle(Form("PMT %d Energy All vs After Veto;Energy (p.e.);Events", pmt + 1));
        double max_pmt = std::max(h_pmt_energy_all_pmt[pmt]->GetMaximum(), h_pmt_energy_veto_pass_pmt[pmt]->GetMaximum());
        h_pmt_energy_all_pmt[pmt]->SetMaximum(max_pmt * 1.2);
        h_pmt_energy_all_pmt[pmt]->SetMinimum(0.1);
        h_pmt_energy_all_pmt[pmt]->Draw();
        h_pmt_energy_veto_pass_pmt[pmt]->Draw("SAME");
        gPad->SetLogy(1);
        TLegend* leg_pmt = new TLegend(0.4, 0.75, 0.6, 0.85);
        leg_pmt->AddEntry(h_pmt_energy_all_pmt[pmt], "All Events", "l");
        leg_pmt->AddEntry(h_pmt_energy_veto_pass_pmt[pmt], "After Veto", "l");
        leg_pmt->Draw();
        c_pmt_individual->Update();
        plotName = Form("%s/PMT%d_Energy_All_vs_VetoPass.png", OUTPUT_DIR.c_str(), pmt + 1);
        c_pmt_individual->SaveAs(plotName.c_str());
        cout << "Saved plot: " << plotName << endl;
        delete leg_pmt;
    }

    // Create combined plot of PMT energy: all events vs after veto
    TCanvas *c_pmt_combined = new TCanvas("c_pmt_combined", "Combined PMT Energy All vs After Veto", 1800, 1200);
    c_pmt_combined->Divide(3, 4, 0, 0); // Same as SPE plot
    int layout[4][3] = {
        {0, 10, 7},
        {2, 6, 3},
        {8, 9, 11},
        {4, 5, 1}
    };
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            int pad = r * 3 + c + 1;
            c_pmt_combined->cd(pad);
            gPad->SetLogy();
            gPad->SetLeftMargin(0.15);
            gPad->SetRightMargin(0.12);
            gPad->SetBottomMargin(0.15);
            gPad->SetTopMargin(0.10);
            int idx = layout[r][c];
            TH1D *h_all = h_pmt_energy_all_pmt[idx];
            TH1D *h_veto = h_pmt_energy_veto_pass_pmt[idx];
            h_all->SetTitle("");
            h_veto->SetTitle("");
            h_all->SetLineColor(kRed);
            h_veto->SetLineColor(kBlue);
            h_all->SetLineWidth(2);
            h_veto->SetLineWidth(2);
            h_all->GetXaxis()->SetTitleSize(0.08);
            h_all->GetYaxis()->SetTitleSize(0.09);
            h_all->GetXaxis()->SetLabelSize(0.04);
            h_all->GetYaxis()->SetLabelSize(0.04);
            h_all->GetYaxis()->SetTitle("Events");
            h_all->GetYaxis()->SetTitleOffset(0.8);
            h_all->GetXaxis()->SetTitle("Energy (p.e.)");
            double max_val = std::max(h_all->GetMaximum(), h_veto->GetMaximum());
            h_all->SetMaximum(max_val * 1.5);
            h_all->SetMinimum(0.5);
            h_all->Draw();
            h_veto->Draw("SAME");
            TLatex *tex = new TLatex();
            tex->SetTextFont(42);
            tex->SetTextSize(0.10);
            tex->SetTextAlign(22);
            tex->SetNDC();
            tex->DrawLatex(0.5, 0.92, Form("PMT %d", idx + 1));
            TLegend *leg = new TLegend(0.65, 0.65, 0.95, 0.75);
            leg->SetBorderSize(0);
            leg->SetTextSize(0.04);
            leg->AddEntry(h_all, "All", "l");
            leg->AddEntry(h_veto, "Veto", "l");
            leg->Draw();
            delete tex;
            delete leg;
        }
    }
    c_pmt_combined->Update();
    plotName = OUTPUT_DIR + "/Combined_PMT_Energy_All_vs_Veto.png";
    c_pmt_combined->SaveAs(plotName.c_str());
    cout << "Saved combined plot: " << plotName << endl;

    // Cleanup
    delete c;
    delete c_pmt_veto;
    delete c_pmt_categories;
    delete c_pmt_individual;
    delete c_pmt_combined;
    delete hMyMuonEnergy;
    delete hMyMichelEnergy;
    delete hMyDtMichel;
    delete hMyEnergyVsDt;
    delete hMyTriggerBits;
    delete hMyNueEnergy;
    delete hMyNueDt;
    delete hMyNuePmtHits;
    delete hMyNueEnergyVsDt;
    delete hMyNueBkgEnergy;
    delete h_pmt_energy_all;
    delete h_pmt_energy_after_veto;
    delete h_pmt_energy_cosmic;
    delete h_pmt_energy_untagged;
    delete h_pmt_energy_tagged;
    delete h_pmt_energy_veto_pass;
    delete h_sipm_untagged;
    delete h_pmt_hits_untagged;
    delete h_dt_untagged;
    delete h_sipm_veto_pass;
    delete h_dt_all;
    delete h_after_veto_trigger2;
    delete h_sipm_cosmic;
    delete h_sipm_tagged;
    for (int i = 0; i < N_PMTS; ++i) {
        delete h_pmt_energy_all_pmt[i];
        delete h_pmt_energy_veto_pass_pmt[i];
    }
    delete leg_veto;
    delete leg_categories;
    return 0;
}
