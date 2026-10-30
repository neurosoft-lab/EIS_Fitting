import argparse
from pathlib import Path

from .studio import (
    DISPLAY,
    STUDIES,
    default_reference,
    fitting_root,
    make_project,
    read_bundle,
    register_reference,
    resolve_result_folder,
    resume_study,
    run_study,
)


def _search(project, args):
    for key in ("starts", "batches", "max_nfev", "workers"):
        project["search"][key] = getattr(args, key)
    return project


def _run_one(app, study, material, data, args, references=None):
    data_root = str(Path(data).resolve()) if data else None
    project = _search(make_project(app, study, material, data_root), args)
    reference = "" if study == "Overlap" else (references or {}).get(material) or default_reference(app, material)
    result = run_study(project, "auto", reference, 3.0)
    if study == "Overlap":
        register_reference(app, result)
    return result


def main(app):
    parser = argparse.ArgumentParser(description="EIS Fitting")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--starts", type=int, default=24)
    common.add_argument("--batches", type=int, default=2)
    common.add_argument("--max-nfev", type=int, default=2000)
    common.add_argument("--workers", type=int, default=4)
    sub = parser.add_subparsers(dest="action", required=True)

    single = sub.add_parser("single", parents=[common])
    single.add_argument("--study", choices=list(STUDIES), required=True)
    single.add_argument("--material", choices=list(DISPLAY.values()), required=True)
    single.add_argument("--data", help="Material folder containing Device1 through Device4")

    batch = sub.add_parser("batch", parents=[common])
    batch.add_argument("--studies", nargs="+", choices=list(STUDIES), default=list(STUDIES))
    batch.add_argument("--materials", nargs="+", choices=list(DISPLAY.values()), default=list(DISPLAY.values()))

    resume = sub.add_parser("resume")
    resume.add_argument("result")
    resume.add_argument("--batches", type=int, default=1)

    args = parser.parse_args()
    if args.action == "single":
        material = next(key for key, value in DISPLAY.items() if value == args.material)
        print(_run_one(app, args.study, material, args.data, args))
        return
    if args.action == "batch":
        references = {}
        last = None
        for study in args.studies:
            for display in args.materials:
                material = next(key for key, value in DISPLAY.items() if value == display)
                last = _run_one(app, study, material, None, args, references)
                if study == "Overlap":
                    references[material] = str(last)
        print(last)
        return

    result = resolve_result_folder(args.result)
    project = read_bundle(result)["project"]
    project["search"]["resume_result"] = str(result)
    project["search"]["batches"] = args.batches
    row = project["datasets"][0]
    project["output_root"] = str(fitting_root(app, row["study"], row["material"]))
    print(resume_study(project))
