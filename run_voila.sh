#!/bin/bash
voila --no-browser main.ipynb --VoilaConfiguration.file_whitelist="['vids.*', 'runs.*']" --Voila.tornado_settings="{'websocket_max_message_size': 209715200}"
