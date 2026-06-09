import os
import yaml
from pathlib import Path
from datetime import datetime

def check_experiment_status(config_dir="configs", output_base="outputs"):
    """检查所有实验的状态"""
    report = []
    
    for config_file in sorted(Path(config_dir).glob("*.yaml")):
        exp_name = config_file.stem
        output_dir = Path(output_base) / exp_name
        
        status = {
            "experiment": exp_name,
            "config_exists": config_file.exists(),
            "output_dir_exists": output_dir.exists(),
            "checkpoints": [],
            "final_model": False,
            "train_log": False,
            "status": "NOT_STARTED"
        }
        
        if output_dir.exists():
            # 检查checkpoints
            checkpoints = list(output_dir.glob("checkpoint-*"))
            status["checkpoints"] = [c.name for c in checkpoints]
            
            # 检查最终模型
            final_dir = output_dir / "checkpoint-final"
            status["final_model"] = final_dir.exists()
            
            # 检查训练日志
            log_file = Path("logs") / f"{exp_name}_train.log"
            status["train_log"] = log_file.exists()
            
            # 判断状态
            if status["final_model"]:
                status["status"] = "COMPLETED"
            elif len(status["checkpoints"]) > 0:
                status["status"] = "IN_PROGRESS"
            else:
                status["status"] = "STARTED_NO_OUTPUT"
        
        report.append(status)
    
    # 生成Markdown报告
    md_lines = [
        "# 实验状态报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 实验ID | 状态 | Checkpoints | 最终模型 | 日志 |",
        "|--------|------|-------------|----------|------|"
    ]
    
    for r in report:
        checkpoints_str = ", ".join(r["checkpoints"]) if r["checkpoints"] else "None"
        final_str = "✅" if r["final_model"] else "❌"
        log_str = "✅" if r["train_log"] else "❌"
        
        md_lines.append(
            f"| {r['experiment']} | {r['status']} | {checkpoints_str} | {final_str} | {log_str} |"
        )
    
    # 统计
    total = len(report)
    completed = sum(1 for r in report if r["status"] == "COMPLETED")
    in_progress = sum(1 for r in report if r["status"] == "IN_PROGRESS")
    not_started = sum(1 for r in report if r["status"] == "NOT_STARTED")
    
    md_lines.extend([
        "",
        "## 统计",
        f"- 总计: {total}",
        f"- 已完成: {completed}",
        f"- 进行中: {in_progress}",
        f"- 未开始: {not_started}",
    ])
    
    report_text = "\n".join(md_lines)
    
    # 保存报告
    with open("experiment_status_report.md", "w") as f:
        f.write(report_text)
    
    print(report_text)
    return report

if __name__ == "__main__":
    check_experiment_status()
