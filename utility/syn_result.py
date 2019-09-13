# Download the most recent folder from remote server to local machine /Downloads

# cd /datadrive/msSum
# "/datadrive/msSum"
# folder_name= ""
# print("tensorboard --logdir  /Users/jcxu/Downloads/{}".format(folder_name))
#
#
# ssh 'intern@20.36.20.141'
# cd /datadrive/msSum
# fd_to_sync=$(ls -t| head -1)
#
# fd_to_sync='tmp_exps_k7pbv0f'
# read var1
#
#
# rsync -rv --exclude='*.th'  --max-size=10m intern@20.36.20.141:/datadrive/msSum/$fd_to_sync   /Users/jcxu/Downloads/
