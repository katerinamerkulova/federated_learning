FROM python:3

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# run the command
CMD ["python", "./distributed.py"]