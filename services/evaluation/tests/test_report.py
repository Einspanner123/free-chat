"""Tests for report generation."""

import os
import sys
import tempfile
import json

import pytest


@pytest.fixture(autouse=True)
def _setup():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class TestReportGenerator:
    def test_markdown_report(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        results = {
            "mmlu": {"accuracy": 0.75, "num_questions": 100, "subjects": {"math": 0.8, "science": 0.7}},
            "gsm8k": {"accuracy": 0.65, "num_problems": 50},
        }
        report = rg.generate_markdown(results, model_name="test-model")
        assert "test-model" in report
        assert "MMLU" in report or "mmlu" in report
        assert "GSM8K" in report or "gsm8k" in report
        assert "75.0%" in report or "0.75" in report
        assert "65.0%" in report or "0.65" in report

    def test_markdown_with_comparison(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        results = {
            "model_a": {"mmlu": {"accuracy": 0.75}, "gsm8k": {"accuracy": 0.65}},
            "model_b": {"mmlu": {"accuracy": 0.80}, "gsm8k": {"accuracy": 0.70}},
        }
        report = rg.generate_comparison_markdown(results)
        assert "model_a" in report
        assert "model_b" in report
        assert "model_a" in report
        assert "MMLU" in report or "mmlu" in report

    def test_html_report(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        results = {"mmlu": {"accuracy": 0.75, "num_questions": 100}}
        report = rg.generate_html(results, model_name="test")
        assert "<html" in report or "<table" in report or "test" in report
        assert "0.75" in report or "75" in report

    def test_json_report(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        results = {"mmlu": {"accuracy": 0.75}}
        report = rg.generate_json(results, model_name="test-model")
        parsed = json.loads(report)
        assert parsed["model"] == "test-model"
        assert parsed["results"]["mmlu"]["accuracy"] == 0.75

    def test_save_report(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        results = {"mmlu": {"accuracy": 0.75}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.md")
            rg.save_report(results, path, format="markdown", model_name="test")
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert len(content) > 0

    def test_save_report_json(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        results = {"mmlu": {"accuracy": 0.75}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            rg.save_report(results, path, format="json", model_name="test")
            assert os.path.exists(path)
            import json
            with open(path) as f:
                data = json.load(f)
            assert data["results"]["mmlu"]["accuracy"] == 0.75

    def test_invalid_format_raises(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        with pytest.raises(ValueError, match="format"):
            rg.save_report({}, "out.txt", format="invalid", model_name="t")


class TestReportVisualization:
    def test_generate_bar_chart(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        data = {"mmlu": 0.75, "gsm8k": 0.65, "humaneval": 0.50}
        chart = rg.generate_bar_chart(data, title="Benchmark Comparison")
        # May return base64 string or file path
        assert chart is not None

    def test_generate_radar_chart(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        data = {
            "model_a": {"mmlu": 0.75, "gsm8k": 0.65},
            "model_b": {"mmlu": 0.80, "gsm8k": 0.70},
        }
        chart = rg.generate_radar_chart(data, title="Model Comparison")
        assert chart is not None

    def test_format_percentage(self):
        from report import ReportGenerator
        rg = ReportGenerator()
        assert rg._format_pct(0.756) == "75.6%"
        assert rg._format_pct(0.0) == "0.0%"
        assert rg._format_pct(1.0) == "100.0%"


class TestExperimentReport:
    def test_document_inference_benchmark(self):
        from report import ExperimentDoc
        doc = ExperimentDoc()
        results = {
            "hf_fp16": {"tps": 8, "latency_ms": 120, "vram_gb": 12.0, "mmlu": 0.652},
            "vllm_fp16": {"tps": 22, "latency_ms": 45, "vram_gb": 11.5, "mmlu": 0.652},
            "vllm_awq": {"tps": 26, "latency_ms": 38, "vram_gb": 4.8, "mmlu": 0.648},
        }
        doc.add_experiment("Inference Engine Comparison", results)
        report = doc.generate()
        assert "Inference Engine" in report
        assert "tps" in report.lower() or "TPS" in report

    def test_document_quantization_comparison(self):
        from report import ExperimentDoc
        doc = ExperimentDoc()
        data = {
            "FP16": {"vram_gb": 12.0, "mmlu": 65.2, "throughput_tps": 8},
            "AWQ 4bit": {"vram_gb": 4.8, "mmlu": 64.8, "throughput_tps": 26},
            "GPTQ 4bit": {"vram_gb": 5.0, "mmlu": 64.5, "throughput_tps": 24},
        }
        doc.add_experiment("Quantization Method Comparison", data)
        report = doc.generate()
        assert "Quantization" in report
        assert "FP16" in report

    def test_document_finetune_comparison(self):
        from report import ExperimentDoc
        doc = ExperimentDoc()
        data = {
            "Base Model": {"ceval": 0.55, "gsm8k": 0.30},
            "LoRA Fine-tuned": {"ceval": 0.62, "gsm8k": 0.35},
            "Full Fine-tuned": {"ceval": 0.65, "gsm8k": 0.37},
        }
        doc.add_experiment("Fine-tuning Comparison", data)
        report = doc.generate()
        assert "Fine-tuning" in report
        assert "LoRA" in report

    def test_document_save(self):
        from report import ExperimentDoc
        doc = ExperimentDoc()
        doc.add_experiment("Test", {"cfg": {"accuracy": 0.5}})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "experiment.md")
            doc.save(path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert len(content) > 0
