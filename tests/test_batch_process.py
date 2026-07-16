import json

from pgixs_python.workflows.batch_process import (
    BatchJob,
    build_job_argv,
    format_batch_summary,
    load_batch_jobs,
    run_batch,
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
