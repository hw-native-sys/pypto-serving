testSuite = ['examples', 'system']

e = executable('engine', ['engine.cpp'], dependencies: [servingBuildDep])
if get_option('buildTests')
	test(
		'engine',
		mpirunExecutable,
		args: [
			'-np', '5',
			'--oversubscribe', e.full_path()
		],
		timeout: 60,
		suite: testSuite,
	)
endif