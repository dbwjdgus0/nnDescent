package types;

public class NodeDistance implements Comparable<NodeDistance>{
    public int id;
    public float dist;

    public NodeDistance(int id, float dist){
        this.id = id;
        this.dist = dist;
    }

    @Override
    public int compareTo(NodeDistance other) {
        if(dist != other.dist)
            return Float.compare(dist, other.dist);
        else
            return Integer.compare(id, other.id);
    }

    @Override
    public String toString() {
        return "NodeVectorDistance{" +
                "id=" + id +
                ", dist=" + dist +
                '}';
    }
}
