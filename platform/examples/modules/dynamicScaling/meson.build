testSuite = ['examples', 'dynamicScaling']

dynamicScaling = executable(
    'dynamicScaling',
    ['dynamicScaling.cpp'],
    dependencies: [servingBuildDep, modulesDependency],
)

if get_option('buildTests')
    test(
        'dynamicScaling',
        mpirunExecutable,
        args: [
            '-np', '1',
            '--oversubscribe',
            dynamicScaling.full_path(),
        ],
        timeout: 60,
        suite: testSuite,
    )
endif
