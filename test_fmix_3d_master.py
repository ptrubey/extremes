import subprocess, itertools as it

cols = list(it.combinations(range(8), 3))

processes = []
for colset in cols:
    processes.append(
        subprocess.Popen(
            ['python', 'test_fmix_3d_slave.py', *[str(x) for x in colset],
            stdin=None, stdout=None, stderr=None,
            )
        )

for process in processes:
    process.wait()

# EOF
