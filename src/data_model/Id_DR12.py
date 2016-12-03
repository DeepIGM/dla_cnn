import Id

class Id_DR12(Id.Id):
    def __init__(self, plate, mjd, fiber, ra=0, dec=0):
        self.plate = plate
        self.mjd = mjd
        self.fiber = fiber
        self.ra = ra
        self.dec = dec

