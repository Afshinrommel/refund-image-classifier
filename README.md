# refund-image-classifier

This repository is a ready-made example for automated image classification prediction using GitHub Actions.

Repository Contents

main.py: A script that processes new images in the caltech_split/val/ directory and appends the results to daily_predictions.csv.

requirements.txt: The required dependencies.

.github/workflows/daily-run.yml: A workflow that runs automatically every night at 21:00 UTC.

Note: The model file caltech_model.h5 is not included in this ZIP (the model is large). Please upload the model file manually into the root of the repository after pushing this repo.
ا
caltech_split/
  └─ val/
      ├─ airplanes/
      │   ├─ img1.jpg
      │   └─ ...
      ├─ motorbikes/
      └─ ...
```

Each subfolder inside caltech_split/val/ is considered one class.
Quick Setup (Local)


Place the caltech_model.h5 model file in the root of the repository.


Arrange the dataset folder according to the structure shown above.


Create a Python environment and install the dependencies:


```bash
pip install -r requirements.txt
```
python main.py
Setup on GitHub
Create a new repository and push the contents of this ZIP.

Add the caltech_model.h5 model file to the root of the repository (use Git LFS if the model is larger than GitHub’s limit).

In the repository, go to the Actions tab; the Daily Batch Prediction workflow will run automatically every night at 21:00 UTC.

After execution, the output file daily_predictions.csv will be available as an artifact in each workflow run.

Manual Workflow Execution
In the Actions tab, click the workflow and press Run workflow (GitHub allows manual execution).

Notes and Recommendations
If the model is very large, use Git LFS or upload the model to Google Drive / S3 and modify the code to download it.

To generate more detailed logs, you can add logging or send output to a database.

If you want to save results daily in a separate table, you can modify main.py so that the output filename includes the date.If you want to store the results on a daily basis in a separate table, main.py can be modified so that the output file name includes the date.









