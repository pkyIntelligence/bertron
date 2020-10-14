import os
import sys
sys.path.insert(0, '/home/wsgi-user/git/bertron')

os.environ['BERTRON_DEVICE'] = 'gpu'

from main_app import app as application
