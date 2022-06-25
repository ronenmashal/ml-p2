FROM ml-project-base:1.0.0
EXPOSE 5000
WORKDIR /app
COPY ./site .
CMD [ "flask", "run", "--host=0.0.0.0"]