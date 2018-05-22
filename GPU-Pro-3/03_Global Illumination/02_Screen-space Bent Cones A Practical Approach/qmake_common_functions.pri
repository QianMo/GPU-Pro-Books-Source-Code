defineReplace(getSourcesFromDir) {
	data =
	filesOfDir =
	win32|win64 {
		filesOfDir = $$system("dir $$1 /B /A:-D")
		for(file, filesOfDir) {
			add = $${1}/$$file
			file = $$split(file, .)
			ending = $$last(file)
			contains(ending, cpp) {
				data += $$add
			}
		}
	}
	return($$data)
}

defineReplace(getHeadersFromDir) {
	data =
	filesOfDir =
	win32|win64 {
		filesOfDir = $$system("dir $$1 /B /A:-D")
		for(file, filesOfDir) {
			add = $${1}/$$file
			file = $$split(file, .)
			ending = $$last(file)
			contains(ending, h) {
				data += $$add
			}
		}
	}
	return($$data)
}

defineReplace(getFilesFromDir) {
	data =
	filesOfDir =
	win32|win64 {
		filesOfDir = $$system("dir $$1 /B /A:-D")
		for(file, filesOfDir) {
			data += $${1}/$$file
		}
	}
	return($$data)
}