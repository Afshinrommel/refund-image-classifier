Automated Refund Item Classification Pipeline
Repository Architecture & Workflow
api_server_two_button_v4.py: This script is responsible for running the client-facing web application. It utilizes Google OAuth client configuration JSONs to perform secure user authentication based on the user's registered Google email address. Once authenticated, the server grants the client permission to upload their items, automatically embedding web-safe execution timestamps into the file-naming convention for database-free metadata tracking.

.github/workflows/main.yml: This configuration manages the automated MLOps batch processing pipeline inside GitHub Actions. Scheduled to execute automatically every 24 hours (nightly batch), it securely connects to the Google Drive staging folder, ingests the newly uploaded customer images, runs the deep learning inference model to identify the refund item categories, and appends the structured results into the centralized predictions.csv log.

dashboard.py (Added Section): The live monitoring and analytics frontend built using Streamlit. It consumes the batch inference results from predictions.csv and offers the following production-grade capabilities:

Dynamic Chronological Sorting: Automatically formats and sorts the prediction table in descending order, ensuring the absolute newest refund requests are always displayed at the very top.

Recent Item Summary: This card automatically catches and displays the newest refund request in the system. It shows the true item, the AI's predicted category, customer info, model confidence, and the exact upload date and time.

Granular Customer Segmentation: Features an interactive sidebar filter that allows administrators to segment the entire dashboard, charts, and image previews by specific customer names.

Automated Cloud Sync & Image Preview: Integrates with gdown to check the local environment and automatically pull missing customer images directly from Google Drive for real-time visual inspection and auditing.

Refund Charts: Shows a clean bar graph that displays the total number of items predicted for each category across the platform.