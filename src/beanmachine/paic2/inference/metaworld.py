import abc

class MetaWorld(metaclass=ABCMeta):
    @abstractmethod
    def print(self):
        raise NotImplementedError()

class RealWorld(MetaWorld):
    def __init__(self, queries: Iterable[RVIdentifier], observations: Dict[RVIdentifier, torch.Tensor]):
        self.python_world = beanmachine.ppl.world.World.initialize_world(queries, observations)

    def print(self):
        print(str(self.python_world))

MetaWorld.register(RealWorld)