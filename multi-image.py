from pathlib import Path


def main():
    maybe_download_and_extract()

    files_dir = Path.cwd() / "newimages"

    for image_path in files_dir.iterdir():
        if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            print(run_inference_on_image(str(image_path)))


if __name__ == "__main__":
    main()
