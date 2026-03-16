#!/bin/bash
#SBATCH --job-name=D2OAnalysis
#SBATCH --output=michel_output.txt
#SBATCH --error=michel_error.txt
#SBATCH --time=2000:00:00
#SBATCH --partition=longjobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G

cd /home/manoja450/FLUX || exit 1

BASE_PATH="/data41/coherent/data/d2o/processedData"

echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Current working directory: $(pwd)"
echo "Starting job at: $(date)"
echo "ROOT version: $(root-config --version)"
echo "ROOT libdir: $(root-config --libdir)"
echo "Library check:"
ldd ./Micheltest_638 | grep -E "Core|MathCore|not found"

if [ ! -f "./Micheltest_638" ]; then
    echo "Error: Micheltest_638 executable not found in $(pwd)"
    exit 1
fi

input_files=()

#for run in {17934..18677}; do   # oct 2024 one month
#for run in {18678..19398}; do   # nov 2024 one month
#for run in {23143..23855}; do   # june2025
#for run in {21698..22417}; do   # apr2025
#for run in {22418..23143}; do   # may2025
#for run in {23143..24617}; do   # june+july 2025
#for run in {17934..24617}; do   # oct 2024 to july 2025
#for run in {15696..15720}; do   # july 2024 24 hr
#for run in {15696..16298}; do   # 24 days lead off july 2024
for run in {17951..17974}; do   # octdata1day2024
    file="${BASE_PATH}/run${run}_processed_v5.root"
    if [ -f "$file" ]; then
        input_files+=("$file")
    else
        echo "Warning: File $file not found, skipping"
    fi
done

if [ ${#input_files[@]} -lt 2 ]; then
    echo "Error: Insufficient input files found (need at least calibration and one data file)"
    echo "Found files:"
    for file in "${input_files[@]}"; do
        echo "  $file"
    done
    exit 1
fi

echo "Input files to process:"
echo "  Calibration file: ${input_files[0]}"
echo "  Data files:"
for ((i=1; i<${#input_files[@]}; i++)); do
    echo "    ${input_files[$i]}"
done

echo "Executing: ./Micheltest_638 ${input_files[*]}"
./Micheltest_638 "${input_files[@]}"
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Error: Micheltest_638 failed with exit code $exit_code"
    exit $exit_code
fi

output_dir=$(ls -td ./AnalysisOutput_* 2>/dev/null | head -n 1)
if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
    echo "Output saved in $(pwd)/$output_dir"
    ls -l "$output_dir/"
else
    echo "Warning: No AnalysisOutput_* directory found"
fi

echo "Job finished at: $(date)"
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
