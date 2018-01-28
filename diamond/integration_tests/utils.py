import subprocess


def run_r_script(r_script):
    R = subprocess.check_output('which Rscript', shell=True).strip()
    subprocess.call(' '.join([R, r_script]), shell=True)
