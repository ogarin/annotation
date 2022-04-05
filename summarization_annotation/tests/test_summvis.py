import sys
import os
from streamlit.cli import main


# This is a test for debugging summvis


os.chdir("..")
if __name__ == '__main__':
    sys.argv = "streamlit run summvis.py".split()
    sys.exit(main())

