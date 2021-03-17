from q2_differential.plugin_setup import plugin
from q2_differential._format import FeatureTensorNetCDFFormat
import xarray as xr
import arviz as az


@plugin.register_transformer
def _100(ff : FeatureTensorNetCDFFormat) -> xr.DataArray:
    return xr.open_dataarray(str(ff))


@plugin.register_transformer
def _101(tensor : xr.DataArray) -> FeatureTensorNetCDFFormat:
    ff = FeatureTensorNetCDFFormat()
    with ff.open() as fh:
        tensor.to_netcdf(fh)
    return ff

@plugin.register_transformer
def _102(ff : FeatureTensorNetCDFFormat) -> xr.Dataset:
    return xr.open_dataarray(str(ff))


@plugin.register_transformer
def _103(tensor : xr.Dataset) -> FeatureTensorNetCDFFormat:
    ff = FeatureTensorNetCDFFormat()
    with ff.open() as fh:
        tensor.to_netcdf(fh)
    return ff


@plugin.register_transformer
def _104(ff : FeatureTensorNetCDFFormat) -> az.InferenceData:
    return xr.from_netcdf(str(ff))


@plugin.register_transformer
def _105(tensor : az.InferenceData) -> FeatureTensorNetCDFFormat:
    ff = FeatureTensorNetCDFFormat()
    with ff.open() as fh:
        tensor.to_netcdf(fh)
    return ff
