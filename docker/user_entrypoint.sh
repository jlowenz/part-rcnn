#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback

USER_ID=${LOCAL_USER_ID:-9001}
GROUP_ID=${LOCAL_GROUP_ID:-9001}

echo "Starting with UID : $USER_ID"
groupadd --gid $GROUP_ID docker_user
useradd --shell /bin/bash -u $USER_ID -p $(echo "user" | openssl passwd -1 -stdin) -o -c "" -m user

export HOME=/home/user

#pip3 install git+git://github.com/jlowenz/keras.git --upgrade --no-deps

chown -R user:docker_user /home/user $HOME/.Xauthority /tmp
exec /usr/sbin/chroot --userspec=user:docker_user --groups=sudo /  "$@"
