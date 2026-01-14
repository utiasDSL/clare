### Workflow

#### Workspace Setup
```bash
# mount project folder
sshfs {your-user-id}@login.ai.lrz.de:/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/ ~/Projects/lrz_server_project/

# mount home folder of user 
sshfs {your-user-id}@login.ai.lrz.de:{your-remote-home-directory} ~/Projects/lrz_server_user/

# unmount them
fusermount -u ~/Projects/lrz_server_project/
fusermount -u ~/Projects/lrz_server_user/
```

Can also add it into `~/.bashrc`
```bash
alias mount_project='sshfs {your-user-id}@login.ai.lrz.de:/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/ ~/Projects/lrz_server_project/'
alias mount_user='sshfs {your-user-id}@login.ai.lrz.de:{your-remote-home-directory} ~/Projects/lrz_server_user/'
alias mount_lrz='mount_project && mount_user'

alias unmount_project='fusermount -u ~/Projects/lrz_server_project/'
alias unmount_user='fusermount -u ~/Projects/lrz_server_user/'
alias unmount_lrz='unmount_project && unmount_user'
```
