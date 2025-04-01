import numpy as np
import cython
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.libc.math import sin, cos


cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    ctypedef struct GEOSCoordSequence

    GEOSContextHandle_t GEOS_init_r() nogil
    void GEOS_finish_r(GEOSContextHandle_t handle) nogil

    GEOSGeometry* GEOSGeom_createLinearRing_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil
    GEOSGeometry* GEOSGeom_createPolygon_r(GEOSContextHandle_t handle, GEOSGeometry* shell, GEOSGeometry** holes, unsigned int nholes) nogil
    void GEOSGeom_destroy_r(GEOSContextHandle_t handle, GEOSGeometry* g) nogil

    GEOSCoordSequence* GEOSCoordSeq_create_r(GEOSContextHandle_t handle, unsigned int size, unsigned int dims) nogil
    void GEOSCoordSeq_destroy_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil
    int GEOSCoordSeq_setX_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double val) nogil
    int GEOSCoordSeq_setY_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double val) nogil

    GEOSGeometry *GEOSIntersection_r(GEOSContextHandle_t handle, const GEOSGeometry* g1, const GEOSGeometry* g2) nogil
    int GEOSArea_r(GEOSContextHandle_t handle, const GEOSGeometry* g, double *area) nogil


@cython.exceptval(check=False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nogil
@cython.cfunc
cdef GEOSGeometry* fov2d_poly(
    handle: GEOSContextHandle_t,
    p: cython.double[::1],
    alpha: cython.double,
    r: cython.double,
    theta: cython.double,
    num_pts: cython.int):
    '''! Create 2D field-of-view polygon.

    @param handle GEOS handle.
    @param p Camera position,
    @param alpha Camera angle, clockwise from positive y axis [rad].
    @param r FoV radius.
    @param theta FoV angle [rad].
    @param num_pts Number of points (≥ 3).
    @return The FoV polygon.
    '''
    cdef cython.Py_ssize_t i
    cdef cython.double a

    cdef GEOSCoordSequence *seq = GEOSCoordSeq_create_r(handle, num_pts + 1, 2)

    GEOSCoordSeq_setX_r(handle, seq, 0, p[0])
    GEOSCoordSeq_setY_r(handle, seq, 0, p[1])

    for i in range(num_pts - 1):
        a = alpha - theta / 2.0 + i / (num_pts - 2.0) * theta
        GEOSCoordSeq_setX_r(handle, seq, 1 + i, p[0] + r * sin(a))
        GEOSCoordSeq_setY_r(handle, seq, 1 + i, p[1] + r * cos(a))

    GEOSCoordSeq_setX_r(handle, seq, num_pts, p[0])
    GEOSCoordSeq_setY_r(handle, seq, num_pts, p[1])

    cdef GEOSGeometry *ring = GEOSGeom_createLinearRing_r(handle, seq)
    return GEOSGeom_createPolygon_r(handle, ring, NULL, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fov2d_overlap_pairs(
    pos: cython.double[:, ::1],
    ang: cython.double[::1],
    r: cython.double = 50.0,
    theta: cython.double = np.radians(90.0),
    num_pts: cython.int = 16,
    mask: cython.bool[:, ::1] = None,
) -> np.ndarray:
    '''! Compute pair-wise 2D field-of-view overlap.

    @param pos Camera positions (n x 2).
    @param ang Camera headings (n).
    @param r FoV radius.
    @param theta FoV angle [rad].
    @param num_pts Number of points in the FoV polygons (≥ 3).
    @return Pair-wise normalized FoV overlap (n x n).
    '''
    assert pos.shape[0] == ang.shape[0]

    cdef cython.Py_ssize_t n = pos.shape[0]

    result = np.zeros((n, n), dtype=np.float32)
    result_view: cython.float[:, ::1] = result

    cdef cython.Py_ssize_t i, j
    cdef cython.double dx, dy, dist_sqr, area
    cdef cython.double fov_area = np.pi * r * r * theta / (2.0 * np.pi)

    cdef GEOSContextHandle_t handle = GEOS_init_r()
    cdef GEOSGeometry **geoms = <GEOSGeometry **>calloc(n, sizeof(GEOSGeometry *))
    cdef GEOSGeometry *intersection

    for i in range(n):
        result_view[i, i] = 1.0
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sqr = dx * dx + dy * dy
            if dist_sqr <= r * r and (mask is None or mask[i, j]):
                if not geoms[i]:
                    geoms[i] = fov2d_poly(handle, pos[i], ang[i], r, theta, num_pts)
                if not geoms[j]:
                    geoms[j] = fov2d_poly(handle, pos[j], ang[j], r, theta, num_pts)
                intersection = GEOSIntersection_r(handle, geoms[i], geoms[j])
                GEOSArea_r(handle, intersection, &area)
                GEOSGeom_destroy_r(handle, intersection)
                result_view[i, j] = result_view[j, i] = area / fov_area

    for i in range(n):
        GEOSGeom_destroy_r(handle, geoms[i])

    free(geoms)

    GEOS_finish_r(handle)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fov2d_overlap_list(
    p1: cython.double[::1],
    a1: cython.double,
    p2: cython.double[:, ::1],
    a2: cython.double[::1],
    r: cython.double = 50.0,
    theta: cython.double = np.radians(90.0),
    num_pts: cython.int = 16
) -> np.ndarray:
    '''! Compute FoV overlap between one camera and a list of other cameras.

    @param p1 Position of first camera (2).
    @param a1 Angle of first camera.
    @param p2 Position of other camera (n x 2).
    @param a2 Angles of other cameras (n).
    @param r FoV radius.
    @param theta FoV angle [rad].
    @param num_pts Number of points in the FoV polygons (≥ 3).
    @return FoV overlaps (n).
    '''
    assert p2.shape[0] == a2.shape[0]

    cdef cython.Py_ssize_t n = p2.shape[0]

    result = np.zeros(n, dtype=np.float32)
    result_view: cython.float[::1] = result

    cdef GEOSContextHandle_t handle = GEOS_init_r()
    cdef GEOSGeometry *geom1 = fov2d_poly(handle, p1, a1, r, theta, num_pts)
    cdef GEOSGeometry *geom2
    cdef GEOSGeometry *intersection

    cdef cython.Py_ssize_t i
    cdef cython.double dx, dy, dist_sqr, area
    cdef cython.double fov_area = np.pi * r * r * theta / (2.0 * np.pi)

    for i in range(n):
        dx = p1[0] - p2[i, 0]
        dy = p1[1] - p2[i, 1]
        dist_sqr = dx * dx + dy * dy
        if dist_sqr <= r * r:
            geom2 = fov2d_poly(handle, p2[i], a2[i], r, theta, num_pts)
            intersection = GEOSIntersection_r(handle, geom1, geom2)
            GEOSArea_r(handle, intersection, &area)
            result_view[i] = area / fov_area
            GEOSGeom_destroy_r(handle, intersection)
            GEOSGeom_destroy_r(handle, geom2)

    GEOSGeom_destroy_r(handle, geom1)
    GEOS_finish_r(handle)

    return result
