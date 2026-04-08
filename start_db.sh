#!/bin/bash
LC_ALL="en_US.UTF-8" pg_ctl -D /opt/homebrew/var/postgresql@16 \
  -l /opt/homebrew/var/postgresql@16/server.log \
  -o "-p 5433" start