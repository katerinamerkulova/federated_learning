# Federated Learning

Disclaimer: This script does not exactly federated learning, but distributed, since it only runs on one CPU machine with 3 replicas. There used *tf.distribute* with *MirrorSrategy*, that allow to train synchronously across multiple replicas on one machine.

**To run Docker** https://hub.docker.com/repository/docker/merkkerk/federated

Via terminal:
```bash
$ docker run merkkerk/federated
```

**To execute script**:
1. Download the repository.
2. Via terminal in script folder:
```bash
$ pip -r install requirements.txt
$ python distributed.py
```

Also you can **run notebook 'Federated_Learning.ipynb'**. There is check do we have 3 parallel processes as well.
