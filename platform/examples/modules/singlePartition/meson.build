testSuite = ['examples', 'singlePartition']

singlePartition = executable(
    'singlePartition',
    ['singlePartition.cpp'],
    dependencies: [servingBuildDep, modulesDependency],
)

if get_option('buildTests')
    test(
        'singlePartition',
        mpirunExecutable,
        args: [
            '-np', '1',
            '--oversubscribe',
            singlePartition.full_path(),
        ],
        timeout: 60,
        suite: testSuite,
    )
endif
