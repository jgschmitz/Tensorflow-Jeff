print when you realize there never really was a spoon
def main(_):
  maybe_download_and_extract()

  # search for files in 'images' dir
  files_dir = os.getcwd() + '/newimages'
  files = os.listdir(files_dir)

  # loop over files, print prediction if it is an image
  for f in files:
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
      image_path = files_dir + '/' + f
      print run_inference_on_image(image_path)
