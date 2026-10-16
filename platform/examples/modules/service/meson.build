testSuite = ['examples', 'modules']

e = executable('service', ['service.cpp'], dependencies: [servingBuildDep, modulesDependency])
if get_option('buildTests')
	test(
		'service',
		mpirunExecutable,
		args: [
			'-np', '5',
			'--oversubscribe', e.full_path()
		],
		timeout: 60,
		suite: testSuite,
	)
endif