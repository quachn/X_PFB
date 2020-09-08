
# Converting lat/long to cartesian
# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
import numpy as np

R = 6371 # radius of the earth

def gps_2_cartesian(lat=None,lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z

# https://gist.github.com/LocalJoost/fdfe2966e5a380957d1c90c462fd1e5c
# Conversion of Geodetic coordinates to the Local Tangent Plane
class GpsUtils:
    # WGS-84 geodetic constants
    a = 6378137           #  WGS-84 Earth semimajor axis (m)
    b = 6356752.3142      #  WGS-84 Earth semiminor axis (m)
    f = (a - b) / a           #  Ellipsoid Flatness
    e_sq = f * (2 - f)    #  Square of Eccentricity
    paris_center_lat = 48.8566
    paris_center_lon = 2.3522

    #  Converts WGS-84 Geodetic point (lat, lon, h) to the
    #  Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z).
    @staticmethod
    def GeodeticToEcef(lat, lon, h):
        #  Convert to radians in notation consistent with the paper:
        lmbda = np.deg2rad(lat)
        phi = np.deg2rad(lon)
        s = np.sin(lmbda)
        N = GpsUtils.a / np.sqrt(1 - GpsUtils.e_sq * s * s)
        sin_lmbda = np.sin(lmbda)
        cos_lmbda = np.cos(lmbda)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x = (h + N) * cos_lmbda * cos_phi
        y = (h + N) * cos_lmbda * sin_phi
        z = (h + (1 - GpsUtils.e_sq) * N) * sin_lmbda
        return x, y, z

    #  Converts the Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z) to
    #  East-North-Up coordinates in a Local Tangent Plane that is centered at the
    #  (WGS-84) Geodetic point (lat0, lon0, h0).
    @staticmethod
    def EcefToEnu(x, y, z,
              lat0, lon0, h0,
              ):
        #  Convert to radians in notation consistent with the paper:
        lmbda = np.deg2rad(lat0)
        phi = np.deg2rad(lon0)
        s = np.sin(lmbda)
        N = GpsUtils.a / np.sqrt(1 - GpsUtils.e_sq * s * s)

        sin_lmbda = np.sin(lmbda)
        cos_lmbda = np.cos(lmbda)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x0 = (h0 + N) * cos_lmbda * cos_phi
        y0 = (h0 + N) * cos_lmbda * sin_phi
        z0 = (h0 + (1 - GpsUtils.e_sq) * N) * sin_lmbda

        xd = x - x0
        yd = y - y0
        zd = z - z0

        #  This is the matrix multiplication
        xEast = -sin_phi * xd + cos_phi * yd
        yNorth = -cos_phi * sin_lmbda * xd - sin_lmbda * sin_phi * yd + cos_lmbda * zd
        zUp = cos_lmbda * cos_phi * xd + cos_lmbda * sin_phi * yd + sin_lmbda * zd
        return xEast, yNorth, zUp

    #  Converts the geodetic WGS-84 coordinated (lat, lon, h) to
    #  East-North-Up coordinates in a Local Tangent Plane that is centered at the
    #  (WGS-84) Geodetic point (lat0, lon0, h0).
    @staticmethod
    def GeodeticToEnu(lat, lon,
                      lat0=None,
                      lon0=None,
                      h=0, h0=0):
        x, y, z = GpsUtils.GeodeticToEcef(lat, lon, h)
        if not lat0:
          lat0=GpsUtils.paris_center_lat
        if not lon0:
          lon0=GpsUtils.paris_center_lon
        xEast, yNorth, zUp = GpsUtils.EcefToEnu(x, y, z, lat0, lon0, h0)
        return xEast, yNorth, zUp

if __name__ == '__main__':
    pass
