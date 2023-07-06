import numpy as np
import pandas as pd
import os

import vodex as vx
import numan as nu
import tifffile as tif
from patchify import patchify, unpatchify

from typing import Union, List, Optional, Tuple, Dict, Any
import numpy.typing as npt
from tqdm.notebook import tqdm, trange

class Voxelizer:
    
    """
    Compute average signal value (of a 4D matrix of fluorescence imaging data: fluorescence intensity at t X z X y X x) across Super Voxels (3D voxels with user-defined size)
    inside a ROI (user-defined mask -multiple ROIs can be provided by providing different integer IDs into the mask).
    Note: the mask should have the same 3D dimentions as a dataset volume and consist of integer values corresponding to voxels of interest 
    for the different ROIs; the 3D Super Voxel size should be no bigger than the dataset volume. 

    Args:
        mask_file: strig of a .tif file with the same 3D dimentions of the dataset volume and integer values correspondent to the voxels of the different ROIs
        superVoxel_size: a 1x3 array [z,y,x] with the dimentions of the Super Voxel (group of voxels to be averaged and considered as single unit)
        roi_id: an integer value correpsondent to the ROI to be analysed (to be patchified in SVs and compute its SV avg); this value is the 
            integer value of the voxels in mask_file correpsondent to the ROI of interest

    Attributes:
        SV_size: a 1x3 array [z,y,x] with the dimentions of the Super Voxel (group of voxels to be averaged and considered as single unit)
        mask: an numpy array with the same 3D dimentions of the dataset volume with integer values correspondent to the voxels of the different ROIs
        padded_mask: zero padded mask to fit integer number of SVs
        all_patches_coordinates: list of [x_min, x_mx, y_min, y_max, z_min, z_max] for each patch in the mask (list of list)
        mask_SVs: list of mask patches values correspondent to the only ROI of interest
        mask_patches_coordinates
    """
    
    def __init__(self, mask_file: str, superVoxel_size: npt.NDArray, roi_id: int):
        self.SV_size = superVoxel_size # order: z,y,x
        # load multi-ROI mask from .tif
        self.mask = self._load_mask(mask_file)
        # zero pad mask by SV size
        self.padded_mask = self._zero_pad_matrix_in_space(self.mask)
        # patchify mask through SV and get locations (get list of patches indeces [i_SV_z, i_SV_y, i_SV_x] for each SV -list of lists-)
        patches_locations = self._get_patches_locations()
        # get coordinates of each patch location
        self.all_patches_coordinates = self._get_patches_coordinates(patches_locations)
        # extract patches related to a specific ROI
        self.mask_SVs, self.mask_patches_coordinates = self._get_ROI_patches(roi_id)

    def _load_mask(self, mask_file: str) -> npt.NDArray: 
        mask = tif.imread(mask_file)
        return mask
    
    def _zero_pad_matrix_in_space(self, data: npt.NDArray) -> npt.NDArray:
        if data.ndim == 3:
            padded = np.zeros((data.shape[0]+self.SV_size[0]-data.shape[0]%self.SV_size[0], data.shape[1]+self.SV_size[1]-data.shape[1]%self.SV_size[1], data.shape[2]+self.SV_size[2]-data.shape[2]%self.SV_size[2]))
            padded[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
        elif data.ndim == 4:
            padded = np.zeros((data.shape[0]*(data.shape[1]+self.SV_size[0]-data.shape[1]%self.SV_size[0])*(data.shape[2]+self.SV_size[1]-data.shape[2]%self.SV_size[1])*(data.shape[3]+self.SV_size[2]-data.shape[3]%self.SV_size[2])))
            padded = np.reshape(padded,(data.shape[0], data.shape[1]+self.SV_size[0]-data.shape[1]%self.SV_size[0], data.shape[2]+self.SV_size[1]-data.shape[2]%self.SV_size[1], data.shape[3]+self.SV_size[2]-data.shape[3]%self.SV_size[2]))
            padded[:, 0:data.shape[1], 0:data.shape[2], 0:data.shape[3]] = data 
        else:
             print('Error: data provided to pad is of inconsistent dimention; should be 3D or 4D matrix (z,y,x) or (t,z,y,x)')
        return padded

    def _get_patches_locations(self) -> List[List[int]]:
        locations = []
        for i_SV_z in np.arange(int(self.padded_mask.shape[0]/self.SV_size[0])):
            for i_SV_y in np.arange(int(self.padded_mask.shape[1]/self.SV_size[1])):
                for i_SV_x in np.arange(int(self.padded_mask.shape[2]/self.SV_size[2])):
                    locations.append([i_SV_z, i_SV_y, i_SV_x])
        return locations
    
    def _get_patches_coordinates(self, patch_locations: List[List[int]]) -> List[List[int]]: #patch_location = [[i_SV_z, i_SV_y, i_SV_x],[],[],...]
        coords = []
        for i_SV in patch_locations:
            i_SV_z, i_SV_y, i_SV_x = i_SV
            z_min = i_SV_z*self.SV_size[0]
            z_max = (i_SV_z+1)*self.SV_size[0]
            y_min = i_SV_y*self.SV_size[1]
            y_max = (i_SV_y+1)*self.SV_size[1]
            x_min = i_SV_x*self.SV_size[2]
            x_max = (i_SV_x+1)*self.SV_size[2]
            coords.append([z_min, z_max, y_min, y_max, x_min, x_max])
        return coords
    
    def _get_ROI_patches(self, roi_id: int) -> Union[List[List[int]], List[List[int]]]:
        # extract patches related to a specific ROI
        mask_SVs = [] # 'good' patches (within mask)
        mask_patches_coordinates = {'z_min':[], 'z_max':[], 'y_min':[], 'y_max':[], 'x_min':[], 'x_max':[]}
        for coords in self.all_patches_coordinates:
            my_patch = self._get_patch(self.padded_mask, coords) == roi_id
            if np.sum(my_patch) > 0:
                        mask_SVs.append(my_patch)
                        z_min, z_max, y_min, y_max, x_min, x_max = coords
                        for key in mask_patches_coordinates.keys():
                             mask_patches_coordinates[key].append(locals()[key])
        return mask_SVs, mask_patches_coordinates
    
    def process_movie(self, experiment: vx.Experiment, batch_size: int) -> Dict: #patchify in SV and calculated avg signal inside each SV
        """
        Calculate mean signal values inside SV within the ROI

        Args:
            experiment: Vodex Experiment object
            batch_size: number of movie volumes (time points) to be loaded and process at a time

        Returns:
            sv_signal: list (SV) of list (time avg) with avg signals for each SV in ROI
        """
        sv_signal = {f"sv_{isv}":[] for isv in range(len(self.mask_SVs))}
        chuncks = experiment.batch_volumes(batch_size, full_only=True, overlap=0)
        for chunck in tqdm(chuncks, desc='Voxelizing chunks'):
            data = experiment.load_volumes(chunck, verbose=False)
            #zero pad dataset
            padded_data = self._zero_pad_matrix_in_space(data)
            for i_patch in range(len(self.mask_SVs)):
                coords = [val[i_patch] for val in self.mask_patches_coordinates.values()]
                # cut dataset around SV
                data_SV = self._get_patch(padded_data, coords)
                # apply mask to SV and get avg signal
                data_SV_avg_on_mask = self._get_avg_masked_signal(data_SV, self.mask_SVs[i_patch])
                sv_signal[f"sv_{i_patch}"].extend(data_SV_avg_on_mask)
        return sv_signal

    def _get_patch(self, data, patch_coord: List[int]) -> npt.NDArray:
        z_min, z_max, y_min, y_max, x_min, x_max = patch_coord
        if data.ndim == 3:
            patch = data[z_min:z_max, y_min:y_max, x_min:x_max]
        elif data.ndim == 4:
            patch = data[:,z_min:z_max, y_min:y_max, x_min:x_max]
        else:
             print('Error: data provided to pad is of inconsistent dimention; should be 3D or 4D matrix (z,y,x) or (t,z,y,x)')
        return patch
    
    def _get_avg_masked_signal(self, data_patched: npt.NDArray, mask_patched: npt.NDArray) -> npt.NDArray:  
        data_mask_SV = data_patched*mask_patched
        avg_SV = np.sum(data_mask_SV, axis=(1,2,3))/np.sum(mask_patched)
        return avg_SV

    def create_signal_df(self, sv_signal: Dict) -> pd.DataFrame:
        """
        Create dataframe with avg raw values for all SV in ROI in time

        Args:
            sv_signal: dictionary of avg signals from voxelizer.process_movie()

        Returns:
            final_cell_df: panda dataframe with avg signals for each SV in ROI in time
        """
        final_cell_df =  pd.DataFrame.from_dict(sv_signal)
        return final_cell_df
    
    def create_coordinates_df(self) -> pd.DataFrame:
        """
        Create dataframe with coordinates for all SV in ROI

        Returns:
            final_coord_df: panda dataframe with coordinates for each SV in ROI
        """
        coords = self.mask_patches_coordinates
        coord_df = pd.DataFrame.from_dict(coords)
        labels = ['sv_'+str(i) for i in range(coord_df.shape[0])]
        coord_df.index=labels
        final_coord_df = coord_df.T
        return final_coord_df
    
    def create_normalized_signal_df(self, sv_signal: Dict, stim_chuncks_dim: int, baseline_volumes: List[int]) -> pd.DataFrame:
        """
        Create dataframe with avg df/f0 values for all SV in ROI in time

        Args:
            sv_signal: dictionary of avg signals from voxelizer.process_movie()
            stim_chuncks_dim: number of time points (volumes) around each stimulus ## we suppose here to use a dataset discountinuous cut around stimuli in windows of stim_chuncks_dim size
            baseline_volumes: list of volumes indices inside stim_chuncks_dim to consider as baseline to get f0
        Returns:
            final_cell_df: panda dataframe with avg df/f0 signals for each SV in ROI in time
        """
        df_array = self._normalize_signal(sv_signal, stim_chuncks_dim, baseline_volumes)
        labels = ['sv_'+str(i) for i in range(df_array.shape[1])]
        final_cell_df = pd.DataFrame(df_array, columns = labels)
        # for i, label in enumerate(labels):
        #     print(i)
        #     final_cell_df = final_cell_df.rename(columns={i:label})
        return final_cell_df
    
    def _normalize_signal(self, sv_signal: Dict, stim_chuncks_dim: int, baseline_volumes: List[int]) -> npt.NDArray:
        raw_array =  np.array(pd.DataFrame.from_dict(sv_signal))
        df_array = np.zeros(raw_array.shape)
        if raw_array.shape[0]%stim_chuncks_dim != 0:
            print('Error: time dimention of dataset does not match stimulus chunck dimention')
        else:
            for cycle in tqdm(range(int(raw_array.shape[0]/stim_chuncks_dim)), desc='Normalization cycles'):
                f0 = np.nanmean(raw_array[np.array(baseline_volumes)+cycle*stim_chuncks_dim,:],axis=0)
                f0[f0==0]=0.000001
                df_array[cycle*stim_chuncks_dim:(cycle*stim_chuncks_dim+stim_chuncks_dim),:] = raw_array[cycle*stim_chuncks_dim:(cycle*stim_chuncks_dim+stim_chuncks_dim),:]/f0
        return df_array
    
    def plot_SVs(self, sv_dict: Dict) -> npt.NDArray:
        """
        Create dataframe with avg df/f0 values for all SV in ROI in time

        Args:
            sv_dict: dictionary with SVs of interest - name(sv_#): value
        Returns:
            plot_matrix: 3D array with original volume size with voxel relative to SV of interest filled with their value (zeros otherwise)
        """
        plot_matrix = np.zeros(self.mask.shape)
        coord_df = self.create_coordinates_df()
        for key in sv_dict.keys():
            z_min, z_max, y_min, y_max, x_min, x_max = coord_df[key]
            plot_matrix[z_min:z_max, y_min:y_max, x_min:x_max] = sv_dict[key]
        return plot_matrix