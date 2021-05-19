from inpainting_time import inpaint
from memory_profiler import profile

def mprofile():
    mlog_name = "mlog/" + "inpainting" + ".mlog"
    with open(mlog_name, "w") as log_file:
        mprofile_inpaint = profile(func=inpaint, stream=log_file)
        mprofile_inpaint()

if __name__ == "__main__":
    mprofile()