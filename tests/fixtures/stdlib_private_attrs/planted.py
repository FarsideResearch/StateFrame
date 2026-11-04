"""AST-only controls; this module is never imported or executed."""

import http.client as hc
import urllib.request as request
from http.client import HTTPSConnection as C
from urllib.request import HTTPSHandler as Handler


class BadHandler(request.HTTPSHandler):
    def https_open(self, request):
        return self._check_hostname


class DerivedHandler(BadHandler):
    def read_parent(self):
        return super()._check_hostname


class TunnelConnection(C):
    def tunnel_host(self):
        return self._tunnel_host


class ImportedHandler(Handler):
    def https_open(self, request):
        return self._check_hostname


class LeakyClassAttr(C):
    @classmethod
    def read_private(cls):
        return cls._missing


class ClientAlias(hc.HTTPConnection):
    def read_private(self):
        return self._missing


class SafeConnection(C):
    def __init__(self):
        self._local = "ours"

    def read_local(self):
        return self._local


class OwnAncestor(C):
    def __init__(self):
        self._ours = "ours"


class OwnDescendant(OwnAncestor):
    def read_local(self):
        return self._ours


class ClassOwned(C):
    _ours = "ours"

    @classmethod
    def read_local(cls):
        return cls._ours


class PlainBase:
    pass


class Plain(PlainBase):
    def read_private(self):
        return self._anything
