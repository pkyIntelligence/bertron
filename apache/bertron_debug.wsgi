import sys
sys.path.insert(0, '/home/wsgi-user/git/bertron')

from main_app import app as application

import types

import cStringIO
import socket
import StringIO

BAD_ITERABLES = [
  cStringIO.InputType,
  socket.SocketType,
  StringIO.StringIO,
  types.FileType,
  types.StringType,
]

class ValidatingMiddleware:

    def __init__(self, application):
        self.__application = application

    def __call__(self, environ, start_response):
        errors = environ['wsgi.errors']

        result = self.__application(environ, start_response)

        value = type(result)
        if value == types.InstanceType:
            value = result.__class__
        if value in BAD_ITERABLES:
            print >> errors, 'BAD ITERABLE RETURNED: ',
            print >> errors, 'URL=%s ' % environ['REQUEST_URI'],
            print >> errors, 'TYPE=%s' % value

        return result

application = ValidatingMiddleware(application)