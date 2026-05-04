import os

def run_audit():
    # Thư mục gốc của project
    PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_artifacts = [
        'models/cyberdetect_mlp_paper_aligned.h5',
        'models/paper_aligned_metadata.json',
        'results/paper_aligned_table3_metrics.csv',
        'results/paper_aligned_ablation_metrics.csv',
        'results/cyberdetect_classification_report.txt',
        'results/cyberdetect_confusion_matrix.csv',
        'results/xai/shap_summary_paper_aligned.png',
        'results/xai/integrated_gradients_paper_aligned.png'
    ]
    
    print(f"Starting audit for project: {PROJECT_PATH}")
    print("-" * 50)
    
    all_ok = True
    for file_path in required_artifacts:
        full_path = os.path.join(PROJECT_PATH, file_path)
        if os.path.exists(full_path):
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            all_ok = False
            
    print("-" * 50)
    if all_ok:
        print("Status: SUCCESS (Found 8/8 required artifacts)")
    else:
        print("Status: MISSING ARTIFACTS")

if __name__ == "__main__":
    run_audit()
