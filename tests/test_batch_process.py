import json

from pgixs_python.workflows.batch_process import (
    BatchJob,
    build_job_argv,
    format_batch_summary,
    load_batch_jobs,
    run_batch,
    save_batch_summary_json,
    validate_batch_jobs,
    write_job_log,
)


def test_load_batch_jobs_from_object(tmp_path):
    batch_file = tmp_path / "batch.json"
    batch_file.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "label": "gixs sample",
                        "workflow": "gixs",
                        "config": "gixs.json",
                        "args": ["--q-bins", "64"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = load_batch_jobs(batch_file)

    assert len(jobs) == 1
    assert jobs[0].workflow == "gixs"
    assert jobs[0].config == "gixs.json"
    assert jobs[0].args == ["--q-bins", "64"]


def test_build_job_argv_adds_shared_manifest_for_workflow():
    job = BatchJob(workflow="trxs", config="trxs.json", args=["--q-bins", "32"], label="sample")

    argv = build_job_argv(job, shared_manifest="manifest.jsonl")

    assert argv == ["--config", "trxs.json", "--q-bins", "32", "--manifest", "manifest.jsonl"]


def test_build_job_argv_does_not_duplicate_manifest():
    job = BatchJob(workflow="gixs", config="gixs.json", args=["--manifest", "job.jsonl"], label="sample")

    argv = build_job_argv(job, shared_manifest="batch.jsonl")

    assert argv == ["--config", "gixs.json", "--manifest", "job.jsonl"]


def test_run_batch_collects_failures_and_continues():
    jobs = [
        BatchJob(workflow="trxs", config="ok.json", label="ok"),
        BatchJob(workflow="gixs", config="bad.json", label="bad"),
    ]

    def ok_runner(argv):
        assert argv == ["--config", "ok.json"]
        return 0

    def fail_runner(argv):
        raise RuntimeError("boom")

    results = run_batch(jobs, runners={"trxs": ok_runner, "gixs": fail_runner, "export-1d": ok_runner})

    assert [result.ok for result in results] == [True, False]
    assert "boom" in results[1].message
    assert "failed: 1" in format_batch_summary(results)


def test_run_batch_handles_system_exit_message_as_failure():
    jobs = [BatchJob(workflow="trxs", config="bad.json", label="bad")]

    def exit_runner(argv):
        raise SystemExit("usage error")

    results = run_batch(jobs, runners={"trxs": exit_runner, "gixs": exit_runner, "export-1d": exit_runner})

    assert results[0].exit_code == 1
    assert results[0].message == "usage error"


def test_validate_batch_jobs_accepts_valid_config(tmp_path):
    config_file = tmp_path / "gixs.json"
    config_file.write_text(json.dumps({"q_bins": 64, "q_scale": "linear"}), encoding="utf-8")
    jobs = [BatchJob(workflow="gixs", config=str(config_file), label="valid")]

    results = validate_batch_jobs(jobs)

    assert results[0].ok
    assert results[0].message == "dry-run ok"


def test_validate_batch_jobs_rejects_unknown_config_key(tmp_path):
    config_file = tmp_path / "gixs.json"
    config_file.write_text(json.dumps({"q_bnis": 64}), encoding="utf-8")
    jobs = [BatchJob(workflow="gixs", config=str(config_file), label="invalid")]

    results = validate_batch_jobs(jobs)

    assert not results[0].ok
    assert "Unknown config option" in results[0].message


def test_run_batch_quiet_captures_output_and_writes_log(tmp_path):
    jobs = [BatchJob(workflow="trxs", config="ok.json", label="quiet job")]

    def noisy_runner(argv):
        print("workflow stdout")
        return 0

    results = run_batch(
        jobs,
        quiet=True,
        log_dir=tmp_path / "logs",
        runners={"trxs": noisy_runner, "gixs": noisy_runner, "export-1d": noisy_runner},
    )

    assert results[0].ok
    assert results[0].stdout.strip() == "workflow stdout"
    assert results[0].log_path is not None
    assert results[0].log_path.exists()
    assert "workflow stdout" in results[0].log_path.read_text(encoding="utf-8")


def test_write_job_log_uses_safe_label(tmp_path):
    job = BatchJob(workflow="trxs", label="bad label:/")
    result = run_batch([job], runners={"trxs": lambda argv: 0, "gixs": lambda argv: 0, "export-1d": lambda argv: 0})[0]

    log_path = write_job_log(tmp_path, 1, result)

    assert log_path.name == "001_bad_label__.log"


def test_save_batch_summary_json(tmp_path):
    job = BatchJob(workflow="trxs", config="ok.json", label="ok")
    result = run_batch([job], runners={"trxs": lambda argv: 0, "gixs": lambda argv: 0, "export-1d": lambda argv: 0})[0]

    out = save_batch_summary_json(tmp_path / "summary.json", [result])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["job_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["jobs"][0]["label"] == "ok"
