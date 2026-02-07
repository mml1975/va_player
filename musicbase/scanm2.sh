#!/bin/bash

exiftool -csv -r -filename -artist -album -title -genre -track -year \
         -bitrate -samplerate -duration -codec \
         -charset id3=cp1251 \
         $1 > $2
