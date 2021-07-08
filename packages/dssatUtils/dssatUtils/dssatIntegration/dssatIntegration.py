from jinja2 import Template
import subprocess
import csv
import numpy as np
import datetime
import copy
import pdb
import os
import subprocess
import tempfile
import shutil
from itertools import cycle
from pathlib import Path
from ..dssatParsing.dssatParsers import DssatIntegrationParser, cast_list_of_strings
import random


class DSSAT():
    """
    The DSSAT class is a python wrapper between the fortran based DSSAT crop-simulator (https://dssat.net/), and the
    gym DssatEnv class. It basically consists in writing input_files, executing DSSAT, and reading back results.
    If DSSAT executable is not located in the user's home, its direction has to be indicated with the dssat_location
    parameter while initializing the class instance. Before deleting a DSSAT instance, you should
    delete the temporary folder by calling self.close().
    """

    def __init__(self, fileX_prefix='UFGA8201', fileX_extension='.MZX', output='HWAM', icdat='82056',
                 planting_date='82057', id_soil='IBMZ910014', random_soil=False, rseed='2150',
                 random_weather=False, nyers=1, soil_path=None, cultivar_bool=False,
                 cultivar_path=None, ingeno='IB0035', cname='McCurdy 84aa', experiment_number=1,
                 dssat_location=None, source_wthr='UFGA.CLI', initial_H2O_factor=.5, random_water_content=False,
                 initial_snh4=.5, initial_sno3=3, lower_p_H2O_factor=.5, upper_p_H2O_factor=.9, additional_data=False,
                 output_context=False, random_planting_date=False, nitro='Y', water='Y', sdate=None, mf=None, seed=1234,
                 mr=0, stateless=True, run_all_experiments=False, dummy_template=False, auxiliary_files_names=None,
                 files_prefix='./'):
        """
        @param fileX_prefix: DSSAT's fileX name with extension
        @type fileX_prefix: string
        @param fileX_extension: DSSAT's fileX extension
        @type fileX_extension: string
        @param output: how the reward in measured, default value 'HWAMS' correspond to the grain yield at maturity
        in kg/ha
        @type output: string
        @param icdat: date of initial condition soil measurement
        @type icdat: string, julian format 'YYDDD'
        @param planting_date: plantin date
        @type planting_date: string, julian format 'YYDDD'
        @param id_soil: soil identifier
        @type id_soil: string
        @param random_soil: if a random soil is generated each time the step function is called
        @type random_soil: bool
        @param rseed: DSSAT internal seed
        @type rseed: integer as string
        @param random_weather: if random weather time series are generated each time the step function is called
        @type random_weather: bool
        @param nyers: the number of years the experiment is ran
        @type nyers: int
        @param soil_path: path to csv file indicating possible soil choices if random_soil is True
        @type soil_path: string
        @param cultivar_bool: if action making consists in choosing a cultivar
        @type cultivar_bool: bool
        @param cultivar_path: path to csv file indicating possible cultivar choices if cultivar_bool is True
        @type cultivar_path: string
        @param ingeno: the cultivar to be grown
        @type ingeno: string
        @param cname: a description to the cultivar to be grown
        @type cname: string
        @param experiment_number: the experiment number in the fileX
        @type experiment_number: integer
        @param dssat_location: path to DSSAT executable
        @type dssat_location: string
        @param source_wthr: name of the WGEN related DSSAT internal file, only to retrieve contextual information
        @type source_wthr: string
        @param initial_H2O_factor: proportion of maximum plant available water when simulation begins
        @type initial_H2O_factor: 0 <= float <= 1
        @param random_water_content: if initial_H2O_factor is randomly chosen at step function call
        @type random_water_content: bool
        @param initial_snh4: soils NH4 in ppm when simulation begins
        @type initial_snh4: float
        @param initial_sno3: soils N03 in ppm when simulation begins
        @type initial_sno3: float
        @param lower_p_H2O_factor: if random_water_content is True, the lower bound the uniform distribution
        @type lower_p_H2O_factor: 0 <= float <= upper_p_H2O_factor
        @param upper_p_H2O_factor: if random_water_content is True, the upper bound the uniform distribution
        @type upper_p_H2O_factor: lower_p_H2O_factor <= float <= 1
        @param additional_data: if contextual input is needed
        @type additional_data: bool
        @param output_context: if contextual input relative to DSSAT's outputs is needed
        @type output_context: bool
        @param random_planting_date: if the planting date is random
        @type random_planting_date: bool
        @param nitro: if nitrogen stress is simulated
        @type nitro: bool
        @param water: if water stress is simulated
        @type water: bool
        @param sdate: starting date of DSSAT simulation
        @type sdate: string, julian format 'YYDDD'
        @param mf: number of the inorganic fertilization section
        @type mf: integer
        @param mr: number of the organic fertilization section
        @type stateless: bool
        @param mr: integer
        @type mr: integer
        @param stateless: if a context should be returned with rewards
        @type stateless: bool
        @param run_all_experiments: if all experiments should be launched with step() method
        @type run_all_experiments: bool
        @param dummy_template: if a fileX in self.files_prefix folder must be directly copied for execution
        @type dummy_template: bool
        @param auxiliary_files_names: names of auxiliary files to copy in DSSAT's temp folder, to be placed in
        self.files_prefix !!
        @type auxiliary_files_names: list
        @param sequential_rseeds: if rseed come from deterministic iterator
        @type sequential_rseeds: bool
        @param seed: int
        @type seed: int
        @param files_prefix: path of all DSSAT related input_files
        @type files_prefix: string
        """
        self.seed = seed
        self.set_seed()
        self.files_prefix = files_prefix
        self.fileX_prefix = fileX_prefix
        self.fileX_extension = fileX_extension
        self.tmp_folder = None
        self.fd = None
        self.tmp_filename = None
        if not dssat_location:
            self.dssat_location = f'{Path.home()}/dssat'
        self.icdat = icdat
        if sdate is None:
            sdate = icdat
        self.sdate = sdate
        self.water = water
        self.nitro = nitro
        self.experiment_number = experiment_number
        if mf is None:
            mf = self.experiment_number
        self.mf = mf
        self.mr = mr
        self.rseed_lower_bound = 0
        self.rseed_upper_bound = 999999
        self.rseed_list = range(self.rseed_upper_bound + 1)
        self.rseed_iterator = None
        self.set_rseed_iterator()
        self.stateless = stateless
        if stateless:
            self.ovvew = 'N'
            self.grout = 'N'
        else:
            self.ovvew = 'Y'
            self.grout = 'Y'
        self.run_all_experiments = run_all_experiments
        self.dummy_template = dummy_template
        self.auxiliary_files_dic = auxiliary_files_names
        self.random_soil = random_soil
        self.random_planting_date = random_planting_date
        if self.random_soil or self.random_planting_date:
            self._planting_date = '00000'
        self._planting_date = copy.deepcopy(planting_date)
        self.pdate = planting_date  # storing fix original planting date in ._planting_date
        self.random_weather = random_weather
        self.random_water_content = random_water_content
        self.initial_shn4 = initial_snh4  # ppm
        self.initial_sno3 = initial_sno3  # ppm
        self.initial_H2O_factor = initial_H2O_factor  # initial soil's PAW fraction
        self.lower_p_H2O_factor = lower_p_H2O_factor
        self.upper_p_H2O_factor = upper_p_H2O_factor
        self.additional_data = additional_data
        if self.random_water_content:
            self.get_random_water_content()
        self.name_soil = None
        self.available_soils = None
        self.soil_path = soil_path
        if random_soil:
            self.load_soils()
        self.id_soil = id_soil
        soil_prefix = id_soil[:2]
        if soil_prefix == 'IB':
            self.source_soil = 'SOIL.SOL'
        else:
            self.source_soil = f'{soil_prefix}.SOL'
        self.output_var = output
        self.source_wthr = source_wthr
        self.soil_template_parameters = None
        self.wthr_file_path = f'{self.dssat_location}/Weather/Climate/{self.source_wthr}'
        self.soil_file_path = f'{self.dssat_location}/Soil/{self.source_soil}'
        self.parser = DssatIntegrationParser(self.soil_file_path, self.wthr_file_path)
        self.wthr_parameters = None
        self.get_wthr_parameters()
        self.weather_out_dic = None
        self.output_context = output_context
        self.additional_data_dic = None
        self.output_dic = None
        self.soil_parameters = None
        self.all_soil_parameters = None
        if not dummy_template:
            self.get_soil_parameters()
            self.get_soil_template_parameters()
        if random_soil:
            self.get_random_soil()  # needs to be executed after parser is loaded, overwrites defaut attribute
        if random_weather:
            self.get_random_weather()
            self.wther = 'W'
        else:
            self.wther = 'M'
            self.rseed = rseed  # default value
        self.nyers = nyers
        self.cultivar_path = cultivar_path
        self.cultivar_descriptions = None
        self.cultivars = None
        self.cultivar_bool = cultivar_bool
        if cultivar_bool:
            self.load_cultivars()
        self.weather_out_cols = ['CLDD', 'DAYLD', 'PRED', 'SRAD', 'TAVD', 'TDYD', 'TGAD', 'TGRD', 'TMND',
                                 'TMXD', 'DOY']
        self.output_context_cols = ['ETCP']
        self.growth_out_dic = None
        self.growth_out_cols = ['DOY', 'GSTD']
        self.additional_data_cols = []
        self.additional_data_cols.extend(self.weather_out_cols)
        self.additional_data_cols.extend([f'DMAXT{max_temp}_{gstd}' for max_temp in [30, 32, 34] for gstd in range(6)])
        self.additional_data_cols.extend([f'DPRE0_{gstd}' for gstd in range(6)])
        self.additional_data_cols.extend([f'PRED_{gstd}' for gstd in range(6)])
        self.additional_data_cols.extend([f'TAVG_{gstd}' for gstd in range(6)])
        self.ingeno = ingeno
        self.cname = cname
        self.make_tmp_folder()
        if auxiliary_files_names is not None:
            self.copy_auxiliary_files(auxiliary_files_names)

    def set_seed(self, seed=None):
        """
        Set a new seed to this program
        @param seed: the random seed
        @type seed: int
        @return: nothing
        @rtype: None
        """
        if seed is None:
            seed = self.seed
        random.seed(seed)
        np.random.seed(seed)

    def set_rseed_iterator(self, rseed_list=None):
        """
        Creates an iterator for DSSAT internal seed called rseed
        :return: nothing
        :rtype: None
        """
        if rseed_list is None:
            rseed_list = self.rseed_list
        np.random.shuffle(np.array(rseed_list))
        self.rseed_list = rseed_list
        self.rseed_iterator = cycle(rseed_list)

    def load_soils(self):
        """
        Loads the csv from self.soil_path, gives the list of available soils if self.random_soil is True
        @return: nothing
        @rtype: None
        """
        with open(f'{self.files_prefix}{self.soil_path}') as csvfile:
            data = csv.DictReader(csvfile)
            all_rows = []
            for row in data:
                all_rows.append(row)
        self.available_soils = all_rows

    def load_cultivars(self):
        """
        Loads the csv from self.cultivar_path, gives the list of available soils if self.cultivar_bool is True
        @return: nothing
        @rtype: None
        """
        cultivars = []
        cultivar_descriptions = []
        with open(f'{self.files_prefix}{self.cultivar_path}') as csvfile:
            data = csv.reader(csvfile)
            for index, row in enumerate(data):
                if index > 0:
                    cultivar, cultivar_description = row
                    cultivars.append(cultivar)
                    cultivar_descriptions.append(cultivar_description)
        cultivar_dic = {cultivar: description for cultivar, description in zip(cultivars, cultivar_descriptions)}
        self.cultivar_descriptions = cultivar_dic
        self.cultivars = cultivars

    def get_random_planting_date(self):
        """
        Generates a random planting date, possibly any day in the year
        @return: nothing
        @rtype: None
        """
        self.pdate = str(np.random.choice(range(1, 366))).zfill(5)

    def get_random_soil(self):
        """
        Chooses a random soil from the ones provided in the csv at self.soil_path
        @return: nothing
        @rtype: None
        """
        if not self.available_soils:
            self.load_soils()
        soil_dic = np.random.choice(self.available_soils)  # one dict per soil
        self.change_soil(soil_dic['id'], soil_dic['source_soil'], soil_dic['name'])

    def get_random_weather(self):
        """
        Changes DSSAT internal seed to generate a new weather series. Warning: the seed has to be in [0, 999999]
        @return: nothing
        @rtype: None
        """
        self.rseed = next(self.rseed_iterator)
        if not self.random_weather:
            self.wther = 'W'

    def get_soil_parameters(self):
        """
        Parses and loads soil parameters for initial conditions initialization
        @return: nothing
        @rtype:None
        """
        self.all_soil_parameters = self.parser.get_dic_from_file('soil')
        self.soil_parameters = self.all_soil_parameters[self.id_soil]

    def get_wthr_parameters(self):
        """
        Parses and loads soil weather parameters for contextual information
        @return: nothing
        @rtype:None
        """
        self.wthr_parameters = self.parser.get_dic_from_file('weather')

    def get_soil_template_parameters(self):
        """
        Prepare the template for soil's initial conditions
        @return: nothing
        @rtype:None
        """
        soil_template_parameters = []
        keys = ['c', 'icbl', 'sh2o', 'snh4', 'sno3']
        for SLB, SDUL, SLLL in zip(self.soil_parameters['SLB'], self.soil_parameters['SDUL'],
                                   self.soil_parameters['SLLL']):
            values = [self.experiment_number, SLB, self.initial_H2O_factor * (SDUL - SLLL) + SLLL, self.initial_shn4,
                      self.initial_sno3]
            soil_template_parameters.append({key: value for key, value in zip(keys, values)})
        self.soil_template_parameters = soil_template_parameters

    def change_soil(self, id_soil, source_soil, name_soil=None):
        """
        Makes the necessary changes if the soil is changed
        @param id_soil: the new soil identifier
        @type id_soil: string
        @param source_soil: the source file name of the new soil
        @type source_soil: string
        @param name_soil: a descripition of the soil
        @type name_soil: string
        @return: nothing
        @rtype:None
        """
        self.id_soil = id_soil
        if name_soil:
            self.name_soil = name_soil
        if source_soil != self.source_soil:
            self.source_soil = source_soil
            self.soil_file_path = f'{self.dssat_location}/Soil/{self.source_soil}'
            self.parser.soil_file_path = self.soil_file_path
            self.get_soil_parameters()
        else:
            self.soil_parameters = self.all_soil_parameters[self.id_soil]
        self.get_soil_template_parameters()

    def get_random_water_content(self):
        """
        Initializes soils water conditions at the beginning of the simulation if random_water_content is True
        @return: nothing
        @rtype:None
        """
        self.initial_H2O_factor = np.random.uniform(self.lower_p_H2O_factor, self.upper_p_H2O_factor)

    def get_context(self):
        """
        Retrieves DSSAT contextual information
        @return: contextual information
        @rtype: dictionnary
        """
        if self.additional_data:
            self.get_additional_data()
        context = {'soil': self.soil_parameters, 'historical_weather': self.wthr_parameters,
                   'additional_data': self.additional_data_dic, 'summary': self.get_output_context()}
        return context

    def get_output(self, path='evaluate.csv'):
        """
        Retrieves DSSAT contextual information
        @return: nothing
        @rtype: None
        """
        output_dic = self.read_data(path)
        self.output_dic = output_dic

    def get_output_context(self):
        """
        Retrieves DSSAT outputs' contextual information
        @return: nothing
        @rtype: None
        """
        if self.output_dic:
            return {feature: self.output_dic[feature] for feature in self.output_context_cols}

    def get_additional_data(self):
        """
        Retrieves DSSAT additional outputs' contextual information
        @return: nothing
        @rtype: None
        """
        weather_out_dic = self.read_data(f'{self.tmp_folder}/weather.csv')
        self.growth_out_dic = self.read_data(f'{self.tmp_folder}/plantgro.csv')
        self.weather_out_dic = weather_out_dic
        self.post_treat_weather_out()

    def read_data(self, path):
        """
        Utility to read csv from indicated path
        @param path: where the csv lies
        @type path: string
        @return: all rows
        @rtype: dictionary
        """
        if os.path.exists(path):  # in case additional_data == True and get_state is called before DSSAT ran
            all_res = []
            with open(path) as csvfile:
                data = csv.reader(csvfile)
                columns = next(data)
                columns_indexes = [index for index, column in enumerate(columns)]
                for row in data:
                    row_values = [row[index] for index in columns_indexes]
                    all_res.append(row_values)
            all_res = np.transpose(all_res)
            all_res = [cast_list_of_strings(res) for res in all_res]
            dic_res = {column: np.array(value) for column, value in zip(columns, all_res)}
            return dic_res

    def post_treat_weather_out(self):
        """
        Treats DSSAT weather outputs
        @return: nothing
        @rtype: None
        """
        if self.weather_out_dic:
            self.additional_data_dic = {}
            for column in self.weather_out_cols:
                self.additional_data_dic[column] = np.mean(self.weather_out_dic[column])
            self.weather_feature_engineering('all', [True] * len(self.weather_out_dic['DOY']))  # for all stages
            DOY_GTSD_WTH_SEL = self.get_growing_stages_sel()
            for gstd, gstd_sel in enumerate(DOY_GTSD_WTH_SEL):
                self.weather_feature_engineering(gstd, gstd_sel)

    def weather_feature_engineering(self, gstd, gstd_sel):
        """
        Statistics on DSSAT weather outputs
        @return: nothing
        @rtype: None
        """
        max_temps = [30, 32, 34]
        for max_temp in max_temps:
            self.additional_data_dic[f'DMAXT{max_temp}_{gstd}'] = (
                    np.array(self.weather_out_dic['TMXD'])[gstd_sel] > max_temp).sum()
        self.additional_data_dic[f'DPRE0_{gstd}'] = (np.array(self.weather_out_dic['PRED'])[gstd_sel] > 0).sum()
        self.additional_data_dic[f'PRED_{gstd}'] = (np.array(self.weather_out_dic['PRED'])[gstd_sel]).sum()
        self.additional_data_dic[f'TAVG_{gstd}'] = (np.array(self.weather_out_dic['TAVD'])[gstd_sel]).sum()

    def get_growing_stages_sel(self):
        """
        Retrieves growing stages' occurrences
        @return: nothing
        @rtype: None
        """
        if self.growth_out_dic:
            GSTD = self.growth_out_dic['GSTD']
            GSTD_unique = range(6)
            DOY_GRO = self.growth_out_dic['DOY']
            DOY_WEATHER = self.weather_out_dic['DOY']
            DOY_GSTD_WTH_SEL = []
            for gstd in GSTD_unique:
                DOY_GSTD = DOY_GRO[GSTD == gstd]
                matches = [True if element in DOY_GSTD else False for element in DOY_WEATHER]
                DOY_GSTD_WTH_SEL.append(matches)
            return DOY_GSTD_WTH_SEL

    def make_template_dic(self):
        """
        Makes the template dictionary to be sent to jinja2 to create the fileX for DSSAT execution
        @return: nothing
        @rtype: None
        """
        dic = {'nyers': self.nyers, 'pdate': self.pdate, 'wther': self.wther, 'id_soil': self.id_soil,
               'rseed': self.rseed, 'ingeno': self.ingeno, 'cname': self.cname,
               'soil_ic_dics': self.soil_template_parameters, 'icdat': self.icdat,
               'water': self.water, 'nitro': self.nitro, 'sdate': self.sdate,
               'ovvew': self.ovvew, 'grout': self.grout, 'mf': self.mf, 'mr': self.mr}
        return dic

    def step(self):
        """
        Runs the full season in DSSAT with current parameters and reads model's outputs, typically the grain yield.
        In particular, runs DSSAT with a filled jinja2 template inside a temporary folder in /tmp. Output format intends
        to match Gym OpenAI conventions.
		@return: model_output
        @rtype: list of one element, [model_output]
		"""
        self.reset_random()
        if self.dummy_template:
            file_X = f'{self.fileX_prefix}{self.fileX_extension}'
            source = f'{self.files_prefix}{file_X}'
            destination = f'{self.tmp_folder}/{file_X}'
            shutil.copyfile(source, destination)
        else:
            template_path = f'{self.files_prefix}{self.fileX_prefix}.jinja2'
            value_dic = self.make_template_dic()  # values to put in template
            with open(template_path) as f_:
                template = Template(f_.read(), trim_blocks=True, lstrip_blocks=True)
            output = template.render(**value_dic)  # outputs the fileX template with correct values
            with open(self.tmp_filename, mode='w') as f_:
                f_.write(output)
        if not self.dummy_template:
            fileX_name_beginning = len(self.tmp_folder) + 1
            file_X = f'{self.tmp_filename[fileX_name_beginning:]}'

        if self.run_all_experiments:
            dssat_command = f'cd {self.tmp_folder} && {self.dssat_location}/run_dssat A ' \
                            f'{file_X}'
        else:
            dssat_command = f'cd {self.tmp_folder} && {self.dssat_location}/run_dssat C ' \
                            f'{file_X} {self.experiment_number}'
        # unfortunately DSSAT fileX file cannot be called with its path but only with its name in local folderA
        with open(os.devnull, "w") as f_:
            subprocess.call(dssat_command, stdout=f_, shell=True)  # puts all shell outputs to trash
        self.get_output(f'{self.tmp_folder}/summary.csv')
        model_output = self.output_dic[self.output_var]
        return np.array(model_output)

    def make_tmp_folder(self):
        """
        Assigns a newly generated temporary folder in /tmp and fileX temporary file.
        @return: nothing
        @rtype: None
        """
        tempfile._Random = random.Random
        self.tmp_folder = tempfile.mkdtemp()
        self.fd, self.tmp_filename = tempfile.mkstemp(suffix=self.fileX_extension,
                                                      prefix=self.tmp_folder + '/')  # creates a temporary .MZX file

    def reset_tmp_folder(self):
        """
        Delete the previous temporary folder if exists, and generates a new temporary folder in /tmp
        and fileX temporary file.
        @return: nothing
        @rtype: None
        """
        self.close()
        self.make_tmp_folder()

    def generate_tmp_folder(self):
        """
        Generates a new temporary folder in /tmp and fileX temporary file for external calls.
        @return: tmpdirname (temporary directory name) , fd (file description of the file inside the temporary
        directory), filename (filname of the file inside the temporary directory)
        @rtype: string, string, string
        """
        tmpdirname = tempfile.mkdtemp()
        fd, filename = tempfile.mkstemp(suffix=self.fileX_extension,
                                        prefix=tmpdirname + '/')
        return tmpdirname, fd, filename

    def reset_random(self):
        """
        Generate a new values for all parameters set to be random
        @return: nothing
        @rtype: None
        """
        if self.random_water_content:  # has to be first in case random_soil == True
            self.get_random_water_content()
        if self.random_soil:
            self.icdat = '00000'  # in case random soil is activated after initialization
            self.get_random_soil()
        if self.random_weather:
            self.get_random_weather()
        if self.random_planting_date:
            self.get_random_planting_date()

    def copy_auxiliary_files(self, names):
        """
        :param names: name of files to DSSAT's temporary folder
        :type names: list
        :return: nothing
        :rtype: None
        """
        for name in names:
            shutil.copyfile(f'{self.files_prefix}{name}', f'{self.tmp_folder}/{name}')

    def close(self):
        """
        Deletes DSSAT's integration temporary folder if exists
        @return: nothing
        @rtype: None
        """
        shutil.rmtree(self.tmp_folder, ignore_errors=True)


if __name__ == '__main__':
    pass
