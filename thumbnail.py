# Presenter Club - Make Great Presentations, Faster
# Copyright (C) 2016 Sam Parkinson <sam@sam.today>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from base64 import b32encode
import os
import tempfile
from PIL import Image
import sys

# Force gtk to use X11 if wayland is running
os.environ['WAYLAND_DISPLAY'] = ''

# Need to setup virtual display BEFORE Gtk is imported
from pyvirtualdisplay import Display
Display(visible=0, size=(640, 640)).start()

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
gi.require_version('WebKit2', '4.0')
from gi.repository import WebKit2
from gi.repository import GLib



class WebView():
    def __init__(self):
        self._win = Gtk.Window()
        self._win.set_default_size(640, 640)
        self._win.connect('destroy', Gtk.main_quit)

        self._webview = WebKit2.WebView()
        self._webview.connect('load-changed', self.__load_changed_cb)

        self._win.add(self._webview)
        self._win.show_all()
        self._win.move(0, 0)

    def load(self, uri, save_fp):
        self._webview.load_uri(uri)
        self._save_fp = save_fp
        Gtk.main()

    def __load_changed_cb(self, webview, load_event):
        if load_event == WebKit2.LoadEvent.FINISHED:
            if not webview.props.uri:
                # What did we actually load?  Nothing.
                return
            # Have to add a timeout so that it actually renders
            GLib.timeout_add(1000, self.__timeout_cb)

    def __timeout_cb(self):
        os.system('import -window root {}'.format(self._save_fp))
        Gtk.main_quit()

    def destory(self):
        self._webview.destroy()
        self._win.hide()
        self._win.destroy()


def gen_thumb(url, save_fp):
    wv = WebView()
    wv.load(url, save_fp)


if __name__ == '__main__':
    name, url, fp = sys.argv
    gen_thumb(url, fp)
