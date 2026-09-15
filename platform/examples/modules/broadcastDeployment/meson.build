testSuite = ['examples', 'modules']

e = executable('broadcastDeployment', ['broadcastDeployment.cpp'], dependencies: [servingBuildDep, modulesDependency])
if get_option('buildTests')
	test(
		'broadcastDeployment',
		mpirunExecutable,
		args: [
			'-np', '5',
			'--oversubscribe', e.full_path(),
			meson.current_source_dir() + '/policy.json',
		],
		timeout: 60,
		suite: testSuite,
	)
endif