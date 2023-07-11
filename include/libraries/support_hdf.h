#pragma once

#include <hdf5.h>

#include <string>
#include <type_traits>

namespace qpc
{
	class	hdf5_file
	{
		protected:
			using size_t	= unsigned long;
			using string	= std::string;	

 			string	name;
			hid_t	file;

			herr_t	status;
			int	mode;	//0=silent,1=noisy

			template<typename>
			struct	undefined_class;	//to warn errors
		public:
			inline	int	close()noexcept
			{
				status	=	H5Fclose(file);
				file	=	0;
				if (status)
				{
					if(mode)printf("HDF5 file %s closed with exception.\n",name.c_str());
					return -2;
				}else
				{
					if(mode)printf("HDF5 file %s closed successfully.\n",name.c_str());
					return 0;
				}
			}//end of close
			inline	int	open(const char* file_name,const char flag, int _mode)noexcept
			{
				name=string(file_name);
				mode=_mode;
				if(file)
				{
					if(int err=close();err)return err;
				}
				switch(flag)
				{
					case 'w': {file=H5Fcreate(file_name,H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);break;}
					case 'r': {file=H5Fopen(file_name,H5F_ACC_RDONLY,H5P_DEFAULT);break;}
					case 'u': {file=H5Fopen(file_name,H5F_ACC_RDWR,H5P_DEFAULT);break;}
					default : {if(mode)printf("invalid flag %c. should be either w, r or u!\n",flag);file=0;return -1;}
				}
				if(file)
				{
					if(mode)printf("HDF5 file %s opened successfully.\n",name.c_str());
					return 0;
				}else 
				{
					if(mode)printf("HDF5 file %s opened with exception.\n",file_name);
					return -2;
				}
			}//end of open

			explicit hdf5_file(const char* file_name,const char flag,int _mode=1):file(0)
			{
				if(open(file_name,flag,_mode))throw;
			}//end of constructor
		
			explicit hdf5_file():name(""),file(0),mode(1)
			{
			}//end of default constructor
		
			virtual	~hdf5_file()noexcept
			{
				if(file)close();
			}//end of destructor

			template<class data_t>
			herr_t	save	(const data_t* data,const size_t datasize,const char* dataname);
			
			template<class data_t>
			herr_t	load	(      data_t* data,const size_t datasize,const char* dataname);

			void	set_mode(int _mode){mode=_mode;}
	};//end of hdf5_file	


	template<class data_t>
	herr_t	hdf5_file::save		(const data_t* data,const size_t datasize,const char* dataname)
	{
		hid_t   dataset,datatype,dataspace;
		hsize_t	dims[1];
		dims[0]=datasize;
		if constexpr(std::is_same_v<data_t,float>)		datatype=H5Tcopy(H5T_NATIVE_FLOAT);
		else if constexpr(std::is_same_v<data_t,double>)	datatype=H5Tcopy(H5T_NATIVE_DOUBLE);
		else if constexpr(std::is_same_v<data_t,long double>)	datatype=H5Tcopy(H5T_NATIVE_LDOUBLE);
		else if constexpr(std::is_same_v<data_t,long>)		datatype=H5Tcopy(H5T_NATIVE_LONG);
		else if constexpr(std::is_same_v<data_t,int>)		datatype=H5Tcopy(H5T_NATIVE_INT);
		else if constexpr(std::is_same_v<data_t,unsigned long>)	datatype=H5Tcopy(H5T_NATIVE_ULONG);
		else if constexpr(std::is_same_v<data_t,unsigned int>)	datatype=H5Tcopy(H5T_NATIVE_UINT);
		else if constexpr(std::is_same_v<data_t,char>)		datatype=H5Tcopy(H5T_NATIVE_CHAR);
		else undefined_class<data_t>{};

		dataspace=	H5Screate_simple(1,dims,NULL);
		dataset  =	H5Dcreate2(file,dataname,datatype,dataspace,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
		status 	 = 	H5Dwrite(dataset,datatype,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
		if(mode)
		{
			if(status>=0)	
			{
				printf("Variable %s saved successfully in file %s.\n",dataname,name.c_str());
			}else
			{
				printf("Variable %s saved unsuccessfully in file %s.\n",dataname,name.c_str());
			}
		}
		status = H5Dclose(dataset);
		status = H5Sclose(dataspace);
		status = H5Tclose(datatype);
		return	status;
	}//end of save

	template<class data_t> 
	herr_t	hdf5_file::load		(data_t* data,const size_t datasize,const char* dataname)
	{
		hid_t   dataset,datatype;
		if constexpr(std::is_same_v<data_t,float>)              datatype=H5Tcopy(H5T_NATIVE_FLOAT);
                else if constexpr(std::is_same_v<data_t,double>)     	datatype=H5Tcopy(H5T_NATIVE_DOUBLE);
                else if constexpr(std::is_same_v<data_t,long double>)   datatype=H5Tcopy(H5T_NATIVE_LDOUBLE);
                else if constexpr(std::is_same_v<data_t,long>)          datatype=H5Tcopy(H5T_NATIVE_LONG);
                else if constexpr(std::is_same_v<data_t,int>)           datatype=H5Tcopy(H5T_NATIVE_INT);
                else if constexpr(std::is_same_v<data_t,unsigned long>) datatype=H5Tcopy(H5T_NATIVE_ULONG);
                else if constexpr(std::is_same_v<data_t,unsigned int>)  datatype=H5Tcopy(H5T_NATIVE_UINT);
                else if constexpr(std::is_same_v<data_t,char>)          datatype=H5Tcopy(H5T_NATIVE_CHAR);
		else undefined_class<data_t>{};
		
		dataset  =      H5Dopen1(file,dataname);
                status   =      H5Dread(dataset,datatype,H5S_ALL,H5S_ALL,H5P_DEFAULT,data);
		if(mode)
		{
			if(status>=0)
                	{
                        	printf("Variable %s loaded successfully in file %s.\n",dataname,name.c_str());
	                }else
        	        {
                	        printf("Variable %s loaded unsuccessfully in file %s.\n",dataname,name.c_str());
                	}
		}
                status = H5Dclose(dataset);
                status = H5Tclose(datatype);
		return	status;
	}//end of load

}//end of qpc
