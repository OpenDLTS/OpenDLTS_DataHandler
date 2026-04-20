from ._typing import *
import numpy as np
def get_data_plot_dict(data_scaling_factor: float, data_type: str, data_x_type: str, condition_type: str) -> TransientDataPlotType:
    # The dictionary is used to store the unit, label and other information of the data plot
    data_plot_dict = {}
    if data_type in ['C','Capacitance','capacitance']:
        data_type_plot = 'Capacitance'
        data_plot_dict['C_unit'] = 'F'
    elif data_type in ['I','Current','current']:
        data_type_plot = 'Current'
        data_plot_dict['C_unit'] = 'A'
    elif data_type in ['V','Voltage','voltage']:
        data_type_plot = 'Voltage'
        data_plot_dict['C_unit'] = 'V'
    elif data_type in ['Vth',r'$\Delta V_{th}$']:
        data_type_plot = r'$\Delta V_{th}$'
        data_plot_dict['C_unit'] = 'V'
    elif data_type in ['R','Resistance',r'$R_{on}$']:
        data_type_plot = 'Resistance'
        data_plot_dict['C_unit'] = 'Ohm'
    else:
        raise ValueError("data_type = 'C' | 'I' | 'V' | 'Vth' | 'R'")
    
    if data_x_type in ['Time','time','t']:
        data_x_type_plot = 'Time'
        data_plot_dict['t_unit'] = '$s$'
    elif data_x_type in ['Frequency','frequency','f']:
        data_x_type_plot = 'Frequency'
        data_plot_dict['t_unit'] = '$Hz$'
    elif data_x_type in ['Voltage','voltage','V']:
        data_x_type_plot = 'Voltage'
        data_plot_dict['t_unit'] = '$V$'
    else:
        raise ValueError("data_x_type = 'Time' | 'Frequency' | 'Voltage'")
    
    if condition_type in ['T','Temperature','temperature']:
        condition_type_plot = 'Temperature'
        data_plot_dict['T_unit'] = '$K$'
    elif condition_type in ['V','Voltage','voltage']:
        condition_type_plot = 'Voltage'
        data_plot_dict['T_unit'] = '$V$'
    else:
        condition_type_plot = condition_type
        data_plot_dict['T_unit'] = '$a.u.$'

    # set plot unit from data_scaling_factor
    unit_list = ['','m',r'\mu','n','p','f']
    unit_scale_factor = 10**np.array([0,3,6,9,12,15])
    unit_index = np.searchsorted(unit_scale_factor,data_scaling_factor)
    data_plot_dict['C_unit'] = unit_list[unit_index]+' '+data_plot_dict['C_unit']
    # Unit of C after scaling
    data_plot_dict['C_unit'] = '$'+data_plot_dict['C_unit']+'$'
    # The scaling factor when plotting the measured value is determined by the unit and the scaling factor of the data itself.
    data_plot_dict['C_plot_factor'] = unit_scale_factor[unit_index]/data_scaling_factor
    data_plot_dict['em_unit'] = '$s^{-1}$'
    data_plot_dict['tau_unit'] = '$s$'
    # Labels when drawing
    data_plot_dict['t_label'] = data_x_type_plot+' '+'['+data_plot_dict['t_unit']+']'
    data_plot_dict['em_label'] = 'Emission Rate'+' '+'['+data_plot_dict['em_unit']+']'
    data_plot_dict['tau_label'] = 'Time Constant'+' '+'['+data_plot_dict['tau_unit']+']'
    data_plot_dict['T_label'] = condition_type_plot+' '+'['+data_plot_dict['T_unit']+']'
    data_plot_dict['C_label'] = data_type_plot+' '+'['+data_plot_dict['C_unit']+']'
    data_plot_dict['DLTS_label'] = 'DLTS Signal'+' '+'['+data_plot_dict['C_unit']+']'
    data_plot_dict['LDLTS_label'] = 'LDLTS Signal'+' '+'['+data_plot_dict['C_unit']+']'
    return data_plot_dict