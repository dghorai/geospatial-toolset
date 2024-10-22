# ubuntu-linux-os
Learning and sharing practical experience of Ubuntu Desktop


### Downloading and Installation of Ubuntu Desktop
1. [Download Ubuntu Desktop](https://ubuntu.com/#download)
2. [Install Ubuntu Desktop](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)


### Create Python Virtual Environment and Install Necessary Packages
- Open Terminal
- sudo apt install python3-virtualenv
- python3 --version
- virtualenv -p python3.10 venv (creating virtual environment for Python version 3.10 as an example)
- source ./venv/bin/activate
- pip3 install spyder (installing spyder using pip as an example)


### Start Spyder
- Open Terminal
- source ./venv/bin/activate
- export QT_QPA_PLATFORM=xcb (this setting needs to do if spyder application not open)
- spyder (type this command to start spyder which is installed in venv)


### [Install, Configure and Use Git](https://linuxhint.com/install-git-ubuntu22-04/)
- sudo apt update
- sudo apt upgrade -y
- git --version (to check if git already installed or not)
- sudo apt install git
- git --version
- git config --global user.name "INSERT YOUR NAME"
- git config --global user.email "INSERT YOUR EMAIL"
- mkdir sample-directory -p
- git init
- ls -a .git
- git config --list (see all config details entered)
- nano ~/.gitconfig (edit if any changes needed; press CTRL and X, then Y then ENTER to exit the nano text editor)
- sudo apt-get install libatk-adaptor ([use GitHub CLI or Git Credential Manager (GCM) to remember credentials instead of using HTTPS from GitHub](https://stackoverflow.com/questions/71522167/git-always-ask-for-username-and-password-in-ubuntu))
- sudo apt install gh (simply run this command to install GitHub CLI)
- gh auth login (select if you want to auth to GitHub.com or GitHub enterprise; select HTTPS; select Yes; it will ask if you want to authenticate with the browser or paste tokens; the one-time code will be displayed in your terminal and your default browser will be opened if you select browser to authenticate; copy and paste them on the tap that is opened)
- git clone <https link> (after authentification done, just do git clone to check if it is working fine or not)
  

### Install GDAL/OGR
- [sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html)
- sudo apt-get update
- sudo apt install libpq-dev
- sudo apt install libpq5=12.2-4
- sudo apt install libpq-dev
- sudo apt install gdal-bin
- sudo apt install libgdal-dev
- export CPLUS_INCLUDE_PATH=/usr/include/gdal
- export C_INCLUDE_PATH=/usr/include/gdal
- [sudo apt install aptitude](https://stackoverflow.com/questions/72887400/install-gdal-on-linux-ubuntu-20-04-4lts-for-python)
- sudo aptitude install libgdal-dev
- apt list --installed | grep "gdal"
- gdalinfo --version
- pip install gdal==3.3.0

  
### Convert DOCX to ODT using Ubuntu Terminal
- libreoffice --headless --convert-to odt *.docx
 
  
### Add Right-Click Functionalities in Ubuntu 22.04
- Add create "New Text Document" option in right-click menu similar to Windows OS
  - Open "Text Editor" app
  - Click on "Save" button
  - Save in "/home/templates" folder in Ubuntu with name like "New Text Document" or anything memorable
  - Now, right-click and you will see "New Document" option activated and under this option you will see "New Text Document" also added and once click on it a new empty text document will be created.
- Reference
  - [Prakash, A. (2023). Add ‘New Document’ Option in Right-Click Menu in Ubuntu.](https://itsfoss.com/add-new-document-option/)
  - [Kaufman, L. (2021). How to Quickly Create a New, Blank Text File on Windows, Mac, and Linux.](https://www.groovypost.com/howto/quickly-create-new-blank-text-file-windows-mac-linux/)
