import subprocess
import os


### 1
# for example you want to download this :
# url:
# video: http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Videos&filename=SubjectSpecific_1.tgz&downloadname=S1
# phpsessid: 7iegacnnh5t3oh8olh46cted85 # already logged in

base_url = 'http://vision.imar.ro/human3.6m/filebrowser.php'
querry = 'download=1&filepath=Poses/D2_Positions&filename=SubjectSpecific_1.tgz' # neglecting ..&downloadname=S1 since you can set it yourself
url = base_url + '?' + querry

dest_dir = os.path.join('/Users/admin/Desktop/temp_del', 'temp.tgz')
PHPSESSID = '7iegacnnh5t3oh8olh46cted85'
subprocess.call([
          'axel',
          '-a',
          '-n', '24',
          '-H', 'COOKIE: PHPSESSID=' + PHPSESSID,
          '-o', 'temp_2d.tgz',
          url])

### extract files

import tarfile

# tarfile.open('xxxx', '@')  # @: r: no compression,    r:gz  gzip compression



