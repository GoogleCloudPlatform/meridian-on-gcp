# Define the base image
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest

# Set the working directory in the container
# This is where your application code will live and where commands will be run from
#WORKDIR /app

# Copy application files into the container
# Replace 'your_app_directory' with the actual directory containing your application code
# For example, if your code is in a 'src' folder in the same directory as your Dockerfile:
# COPY src/ .
# If you just have a requirements.txt and a Python script:
# COPY requirements.txt .
# COPY your_script.py .

# Install any needed Python packages
# If you have a requirements.txt file:
# RUN pip install --no-cache-dir -r requirements.txt
# If you need to install specific packages directly:
RUN pip install --no-cache-dir --upgrade google-meridian[colab,and-cuda]

# Set environment variables (if needed)
# For example:
# ENV MY_VARIABLE="my_value"
# ENV PYTHONUNBUFFERED=1

# Expose any ports your application might use (if it's a web service, for example)
# For example, if your application listens on port 8080:
# EXPOSE 8080

# Define the command to run your application
# This is what will be executed when the container starts
# Replace 'your_script.py' with the entrypoint to your application
# For example, to run a Python script:
# CMD ["python", "your_script.py"]
# Or if your application has a specific command to start:
# CMD ["your_application_command"]
