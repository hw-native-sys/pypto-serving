testSuite = ['examples', 'modules']

e = executable('coordinatorReplica', ['coordinatorReplica.cpp'], dependencies: [servingBuildDep, modulesDependency])
if get_option('buildTests')
	test(
		'coordinatorReplica',
		mpirunExecutable,
		args: [
			'-np', '2',
			'--oversubscribe', e.full_path(),
			meson.current_source_dir() + '/policy.json',
		],
		timeout: 60,
		suite: testSuite,
	)
endif
