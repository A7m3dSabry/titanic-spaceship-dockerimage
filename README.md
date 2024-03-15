# Titanic Spaceship Model
## Description
This repo has a model used to solve famous kaggle competition [Titanc Spaceship](https://www.kaggle.com/competitions/spaceship-titanic)
the model also containd in a docker image published on [docker hub](https://hub.docker.com/r/a7m3dsabry/titanic_task)


## How to Use
### installing docker
you can follow their [documentation](https://docs.docker.com/get-docker/) it's pretty easy to follow

### pulling the container

pull the container using
'docker pull a7m3dsabry/titanic_task'

### run container

  * train a model
    `sudo docker run -it --entrypoint /bin/sh test -v {HostFolderPath}:/{MountFolderPath}'  -c 'python3 /app/train.py {MountFolderPath}/{TrainCsvFile}'`
  * test the model
    `'sudo docker run -it --entrypoint /bin/sh test -v {HostFolderPath}:/{MountFolderPath}'  -c 'python3 /app/test.py {MountFolderPath}/{TrainCsvFile}'`

### the output ?
after training or testing you will find `train.txt` or `test.txt` on the mounted folder specified in the run command
