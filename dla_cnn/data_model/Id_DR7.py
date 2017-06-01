from Id import Id


class Id_DR7(Id):
    def __init__(self, plate, fiber, ra=0, dec=0):
        super(Id_DR7,self).__init__()
        self.plate = plate
        self.fiber = fiber
        self.ra = ra
        self.dec = dec

    def id_string(self):
        return "%05d-%05d" % (self.plate, self.fiber)

