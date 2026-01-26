"""
A simple file argument handler which checks format.

Example usage: 
				Call in x.py 					-->			Return 			<--		File Argument Call
	file_arguments.get(str, int, float, fill_empties_with_none=True) 	--> ("abc", 1, 3.14) 				<-- "python x.py abc 1 3.14" was called.
	file_arguments.get(str, int, float, fill_empties_with_none=True) 	--> ("abc", None, None)				<-- "python x.py abc" was called.
	file_arguments.get(i=str, o=str, require_options=True) 			--> {i="in", o="out"}				<-- "python x.py i in -o out" was called.
	file_arguments.get(i=str, o=str, require_options=True) 			--> {i=("in", "in2"), o=("out", "out2")}	<-- "python x.py i in in1 -o out out2" was called.
	file_arguments.get(int, i=str, o=str) 					--> (1), {i="in"}				<-- "python x.py 1 -i in" was called.
"""

import sys


class ArgumentForm:
	"""Class that handles the form for a given argument, or arguments with repeated form."""
	
	class IncorrectArgumentForm(Exception):
		pass
	
	def __init__(self, *args, name=None):
		"""
		Enter at least one of the following unordered arguments: 
		    > type: The type of the argument (e.g. int). None means all types are allowed. Identified by the type "type".
		    > num: The number of arguments (e.g. 3). None means any number of arguments are allowed. Identified by the type "int".
		Also, an optional "name" keyword parameter is used to identify the argument form from others.
		"""
		if len(args) <= 0: raise TypeError('There must be at least one format argument.')
		if len(args) > 2: raise TypeError('There must be at most two format arguments.')
		self.name = None if name is None else str(name)  # name must be a string (or something that can easily be turned into one)!

		self.type = None	# Type determines the type of the argument. None means all types are allowed.
		self.num = None		# Integer determines the number of arguments this represents. None means any number of arguments are allowed. 
		for arg in args:
			if type(arg) is type:
				if self.type is None:
					self.type = arg
				else:
					raise TypeError('There must be at most one "type" type argument.')
			elif type(arg) is int:
				if self.num is None:
					self.num = arg
				else:
					raise TypeError('There must be at most one "int" type argument.')
			else:
				raise TypeError('There must be only "type" or "int" type arguments entered.')
		if self.type is None and self.num is None:
			raise TypeError('There must be at least one type or integer format argument.')
			
	def check_form(self, *args, fill_empties_with_none=False):
		"""Checks if the entered argument(s) is in the correct form."""
		args = list(args)
		num_empties = 0
		if self.num is not None:
			# check number of arguments
			if len(args) < self.num:
				if fill_empties_with_none:
					num_empties = self.num - len(args)
				else:
					raise ArgumentForm.IncorrectArgumentForm('Not enough arguments for option "{}". Requires {} argument(s) and instead {} were entered.'.format(self.name, self.num, len(args)))
			elif len(args) > self.num:
				raise ArgumentForm.IncorrectArgumentForm('Too many arguments for option "{}". Requires {} argument(s) and instead {} were entered.'.format(self.name, self.num, len(args)))

		# check if each argument has the right type
		for i in range(len(args)):
			try:
				args[i] = self.type(args[i])
			except:
				raise ArgumentForm.IncorrectArgumentForm('Wrong type entered in argument "{}". It must have type: {}.'.format(args[i], self.type))
		
		result = tuple(list(args) + [None]*num_empties)
		return result[0] if len(result) == 1 else result  # add any empties as None at the end


class FullArgumentForm:
	"""Class that handles "full" argument forms potentially consisting of multiple types and of option flags."""
	def __init__(self, *form):
		"""The arguments must have type ArgumentForm. Option ArgumentForms are identified by their name property. """
		try:
			for f in form:
				if type(f) is not ArgumentForm:
					raise TypeError
		except TypeError:
			raise TypeError("Form must have elements of type ArgumentForm.")

		option_names = [f.name for f in form if f.name is not None]
		if len(set(option_names)) != len(option_names): 
			raise TypeError('There must not be any repeated options in the form definition.')
		if None in option_names and option_names.index(None) == list(reversed(option_names)).index(None):
			raise TypeError('All non-options must be at the beginning of form.')

		for f in form:
			if f.num != 1 and f.name == None: 
				raise NotImplementedError('At the moment, multiple argument ArgumentForms are only supported for options.')

		self.form = form

	def check_form(self, *args, fill_empties_with_none=False, require_options=False):
		"""Checks if the entered argument(s) is in the correct form."""
		# split args into main forms and option forms
		form_option_names = [f.name for f in self.form if f.name is not None]  # options are identified by their name property
		arg_option_names = []
		option_indices = [-1]  # "option" at -1 ==> begins at 0
		for i, arg in enumerate(args):  # for each argument
			if str(arg) in form_option_names and arg is not None:  # is this an option flag?
				option_indices.append(i)  # if so, append where it is in the arugment
				arg_option_names.append(str(arg))  # and save the option name that it is also
		option_indices.append(len(args))
		
		if len(set(arg_option_names)) != len(arg_option_names):
			raise ArgumentForm.IncorrectArgumentForm('There must not be more than one of the same option flag.')

		if require_options and len(form_option_names) != len(arg_option_names):
			raise ArgumentForm.IncorrectArgumentForm('Not all option flags are present, but they are required to.')
		
		split_args = [args[option_indices[i]+1:option_indices[i+1]] for i in range(len(option_indices) - 1)]

		# first split (always non-options)
		first_args = split_args[0]
		first_form = [f for f in self.form if f.name is None]
		first_form_num = sum([f.num for f in first_form])  # assumes that f.num != None

		num_empties = 0
		if len(first_args) < first_form_num: # check number of arguments
			if fill_empties_with_none:
				num_empties = first_form_num - len(first_args)
			else:
				raise ArgumentForm.IncorrectArgumentForm("Not enough required arguments. Requires {} argument(s) and instead {} were entered.".format(first_form_num, len(first_args)))
		elif len(first_args) > first_form_num:
			raise ArgumentForm.IncorrectArgumentForm("Too many required arguments. Requires {} argument(s) and instead {} were entered.".format(first_form_num, len(first_args)))

		return_tuple = []
		for arg, argform in zip(first_args, first_form):
			return_tuple.append(argform.check_form(arg))
		return_tuple = tuple(return_tuple + [None]*num_empties)  # add any empties as None at the end

		# do all the options now
		return_dict = {}
		for option_name, split_args in zip(arg_option_names, split_args[1:]):
			split_form = [f for f in self.form if f.name == option_name]
			if len(split_args) != 0:
				arg = split_args
				assert len(split_form) == 1  # this is just assumed right below
				return_dict[option_name] = split_form[0].check_form(*arg, fill_empties_with_none=fill_empties_with_none)
			else:
				if fill_empties_with_none or split_form[0].num is None:
					return_dict[option_name] = None
				else:
					raise ArgumentForm.IncorrectArgumentForm("Not enough arguments for option {}. Requires {} argument(s) and instead {} were entered.".format(option_name, split_form[0].num, len(split_args)))

		if len(return_dict) == 0:
			return return_tuple
		elif len(return_tuple) == 0:
			return return_dict
		else:
			return return_tuple, return_dict


class FileArguments:
	"""Class that handles getting and checking the arguments to a python file."""

	def __init__(self, form=None, fill_empties_with_none=False, require_options=False,  check=False, args=None):
		"""Initializes the contraints on the arguments. Form must have type FullArgumentForm."""
		if form is not None and type(form) is not FullArgumentForm:
			raise TypeError("Form must be None or have type FullArgumentForm.")
		if args is not None:
			try:
				list(args)
			except TypeError:
				raise TypeError('Manually entering arguments requires that they are iterable.')

		self.form = form
		self.fill_empties_with_none = fill_empties_with_none
		self.require_options = require_options

		self.args = args
		self._args_checked = False

		if check: 
			self.get_args()
	
	def _get_args(self):
		"""Gets the arguments without checking them"""
		if self.args is None:
			self.args = sys.argv[1:]
		return self.args

	def _check_args(self):
		"""Checks the arguments for if they are compatible with the desired contraints"""
		assert self.args is not None, 'The arguments must be defined before they are checked.'
		if not self._args_checked:
			if self.form is not None:
				self.args = self.form.check_form(*self.args, fill_empties_with_none=self.fill_empties_with_none, require_options=self.require_options)
		self._args_checked = True
		
	def get_args(self):
		"""Gets the arguments and checks if they are of the right form."""
		self._get_args()
		self._check_args()
		return self.args


def get_file_arguments(*form, fill_empties_with_none=False, require_options=False, option_prefix='-', **options_form):
	"""Shortens getting the file arguments. This is the main function to call."""
	# construct the form objects
	form = list(form)
	for i, f in enumerate(form):
		try:
			(*f,)
			raise NotImplementedError('Currently no support for fancy formatting in non-options form.')
		except TypeError: # if not iterable
			f = (f,1)
		form[i] = ArgumentForm(*f)
	for k, v in options_form.items():
		try:
			(*v,)
		except TypeError: # if not iterable
			v = (v,)
		options_form[k] = ArgumentForm(*v, name=option_prefix+k) # overwrite previous one
		
	form = list(form) + list(options_form.values())
	fullform = FullArgumentForm(*form)

	# find, check and return arguments
	return FileArguments(fullform, fill_empties_with_none=fill_empties_with_none, require_options=require_options, check=True).args


get=get_file_arguments  # alias for cleaner look

