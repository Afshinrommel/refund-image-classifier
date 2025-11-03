import requests

url = "http://127.0.0.1:5000/predict"
with open(r"C:/Un/Semster 4/Book/model to production/refund-image-classifier/test_images/test2.jpg", "rb") as f:
    response = requests.post(url, files={"file": ("test2.jpg", f)})
print(response.json())
