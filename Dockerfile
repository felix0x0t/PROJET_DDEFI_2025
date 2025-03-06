# Step 1: Use a base image with Python
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the required files into the container
COPY requirements.txt /app/

# Step 4: Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the Dash application file (dashboard.py) into the container
COPY dashboard.py /app/

# Step 6: Expose the port where the Dash app will run (default is 8050)
EXPOSE 8050

# Step 7: Define the command to run the Dash app
CMD ["python", "dashboard.py"]
