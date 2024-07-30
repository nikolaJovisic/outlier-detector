#/bin/bash

./inspect.sh outliers | xargs -d '\n' ./to_png.sh
